import datetime as dt
from dataclasses import dataclass
import multiprocessing as mp

import pandas as pd
import numpy as np
import numba


@dataclass
class ForecastScenario:
    """
    configuration holder for forecasting scenario
    """

    forecast_date: dt.datetime
    forecast_months: int
    product_1_smm: pd.Series
    product_2_smm: pd.Series
    product_1_mdr: pd.Series
    product_2_mdr: pd.Series
    boe_base_rate: float
    recovery_curve: pd.Series
    assumed_mortgage_tenor: int
    interest_only_mortgages: bool = True
    

#@numba.njit
def pmt_numba_array(rate, nper, pv, fv =0, when =0):
    """
    numba-appropriate PMT implementation rather than relying
    on the slower one from numpy-financial

    see here: https://github.com/numpy/numpy-financial/issues/36
    """
    c = (1+rate)**nper
    return np.where(nper == 0, np.nan,
                    np.where(rate ==0, -(fv + pv) /nper ,
                             (-pv *c  - fv) * rate / ( (c - 1) *( 1 + rate * when ) ) ) )


#@numba.njit
def compute_survival_pct_numba(expected_balance_post_defaults: np.ndarray, cashflows: np.ndarray) -> np.ndarray:
    """
    numba-friendly way to get the survival pct
    """
    
    survival_pct = np.zeros_like(expected_balance_post_defaults)
    for i in range(expected_balance_post_defaults.shape[0]):
        if cashflows[i, 0] != 0:
            survival_pct[i] = expected_balance_post_defaults[i] / cashflows[i, 0]
    return survival_pct


#@numba.njit
def do_cashflows_timestep(
        balance: np.ndarray,
        remaining_term: np.ndarray,
        time_past_reversion: np.ndarray,
        fixed_pre_reversion_rate: np.ndarray,
        post_reversion_margin: np.ndarray,
        is_interest_only: np.ndarray,
        base_rate: float
) -> np.ndarray:
    """
    single time step for cashflows
    """
    
    # calculate required payments based on loan type, reversion etc
    interest_rate = np.where(time_past_reversion > 0, base_rate + post_reversion_margin, fixed_pre_reversion_rate)
    target = np.where(is_interest_only == 1, balance, 0)
    # payment = npf.pmt(interest_rate / 12, remaining_term, -balance, target)
    payment = pmt_numba_array(interest_rate / 12, remaining_term, -balance, target)
    
    # split payment into interest and principal, and decrement the loan balances accordingly
    scheduled_interest = balance * interest_rate * 1 / 12
    scheduled_principal = payment - scheduled_interest

    # if last month of interest only mortgage then need to repay the full balance
    is_final_month = remaining_term == 1
    interest_only = is_interest_only == 1
    principal_balloon = np.where(is_final_month & interest_only, balance, 0)
    total_scheduled_principal = scheduled_principal + principal_balloon
    closing_balance = balance - total_scheduled_principal
    result = np.empty((balance.shape[0], 5), dtype=balance.dtype)

    # Assign each array to the corresponding column in the result array
    result[:, 0] = balance
    result[:, 1] = payment
    result[:, 2] = scheduled_interest
    result[:, 3] = total_scheduled_principal
    result[:, 4] = closing_balance

    return result


#@numba.njit
def do_expected_cashflows_timestep(
        expected_performing_balance: np.ndarray,
        cashflows: np.ndarray,
        cpr: np.ndarray,
        cdr: np.ndarray,
) -> np.ndarray:
    """
    single time step in model
    """
    
    # get expected defaults and adjust payments accordingly
    defaults = (1 - np.pow(1 - cdr, (1/12))) * expected_performing_balance
    expected_balance_post_defaults = expected_performing_balance - defaults
    survival_pct = np.divide(
        expected_balance_post_defaults, 
        cashflows[:, 0], 
        out=np.zeros_like(expected_balance_post_defaults), 
        where=cashflows[:, 0] != 0
    )

    # if using numba then need to use the below function as np.divide not supported with above args
    # survival_pct = compute_survival_pct_numba(expected_balance_post_defaults, cashflows)
    expected_payment = survival_pct * cashflows[:, 1]
    expected_interest_payment  = survival_pct * cashflows[:, 2]
    expected_principal_payment = survival_pct * cashflows[:, 3]
    
    # adjust for expected pre-payments
    expected_balance_pre_period_repays = expected_balance_post_defaults - expected_principal_payment
    expected_prepayments = (1 - np.pow(1 - cpr, (1 / 12))) * expected_balance_pre_period_repays
    expected_closing_balance = expected_balance_pre_period_repays - expected_prepayments

    # pre-allocate array and return results
    result = np.empty((expected_performing_balance.shape[0], 9), dtype=expected_performing_balance.dtype)
    result[:, 0] = expected_performing_balance
    result[:, 1] = defaults
    result[:, 2] = expected_balance_post_defaults
    result[:, 3] = expected_payment
    result[:, 4] = expected_interest_payment
    result[:, 5] = expected_principal_payment
    result[:, 6] = expected_balance_pre_period_repays
    result[:, 7] = expected_prepayments
    result[:, 8] = expected_closing_balance

    return result


#@numba.njit
def do_expected_defaults_recoveries_timestep(
        expected_default_balance: np.ndarray,
        expected_cashflows: np.ndarray,
        recovery_curve: np.ndarray,
        loss_curve: np.ndarray,
        default_forecasts: np.ndarray
) -> np.ndarray:
    """
    calculate expected recoveries and losses from 
    the defaults
    """

    # get expected defaults and the lagged loss/recoveries
    expected_new_defaults = expected_cashflows[:, 1]
    
    # take the product of the historic defaults with our loss and
    # recovery curves to get total loss/recovery per loan
    historic_defaults = default_forecasts[:, 1, :]
    expected_recovery = (historic_defaults @ recovery_curve.T).reshape(-1)  # ensure output 1D
    expected_loss = (historic_defaults @ loss_curve.T).reshape(-1)  # ensure output 1D

    # add new defaults to balance, and remove anything now marked as recovered/lost
    expected_closing_default_balance = expected_default_balance + expected_new_defaults - expected_recovery - expected_loss

    # pre-allocate an array and return the results
    result = np.empty((expected_default_balance.shape[0], 5), dtype=expected_default_balance.dtype)
    result[:, 0] = expected_default_balance
    result[:, 1] = expected_new_defaults
    result[:, 2] = expected_recovery
    result[:, 3] = expected_loss
    result[:, 4] = expected_closing_default_balance
    return result


#@numba.njit
def do_full_timestep(
        balance: np.ndarray,
        expected_performing_balance: np.ndarray,
        remaining_term: np.ndarray,
        time_past_reversion: np.ndarray,
        fixed_pre_reversion_rate: np.ndarray,
        post_reversion_margin: np.ndarray,
        is_interest_only: np.ndarray,
        expected_default_balance: np.ndarray,
        cpr: np.ndarray,
        cdr: np.ndarray,
        recovery_curve: np.ndarray,
        loss_curve: np.ndarray,
        base_rate: float,
        default_forecasts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TBC
    """

    cashflows = do_cashflows_timestep(
        balance,
        remaining_term,
        time_past_reversion,
        fixed_pre_reversion_rate,
        post_reversion_margin,
        is_interest_only,
        base_rate
    )

    expected_cashflows = do_expected_cashflows_timestep(
        expected_performing_balance=expected_performing_balance,
        cashflows=cashflows,
        cpr=cpr,
        cdr=cdr,
    )

    default_cashflows = do_expected_defaults_recoveries_timestep(
        expected_default_balance=expected_default_balance,
        expected_cashflows=expected_cashflows,
        recovery_curve=recovery_curve,
        loss_curve=loss_curve,
        default_forecasts=default_forecasts
    )
    
    return cashflows, expected_cashflows, default_cashflows


def _run_forecast_internal(
        n_months: int,
        loans: np.ndarray,
        cpr: np.ndarray,
        cdr: np.ndarray,
        recovery_curve: np.ndarray,
        base_rate: np.ndarray
) -> np.ndarray:
    """
    Run forecast for the given loan portfolio

    Args:
    n_months (int): number of months into the future to run cashflow 
    forecasts for
    loans (np.ndarray): m x 6 array defining loans, where the columns in order
    are:
        - the initial balance
        - the remaining term
        - the time past reversion point
        - the loan's fixed pre-reversion rate
        - the loan's post-reversion margin
        - binary column for is interest only (1) or not (0)
    cpr (np.ndarray): m x n_months array giving the conditional prepayment rate for each 
    mortgage in each forecast month
    cdr (np.ndarray): m x n_months array giving the conditional default rate for each
    mortgage in each forecast month
    recovery_curve (np.ndarray): 1 x k month array showing % of default exposure recovered in each month after
    default
    base_rate (float): 1 x n_months array of BoE base rates projected for each forecast month
    """

    # pre-allocate arrays for our output - each of these is of the
    # shape number of loans x number of fields x number of forecast months
    n_loans = loans.shape[0]
    cashflow_schedule = np.empty(shape=(n_loans, 5, n_months))
    cashflow_forecasts = np.empty(shape=(n_loans, 9, n_months))
    default_forecasts = np.empty(shape=(n_loans, 5, n_months))
    
    # need to pad out the defaults so that we have enough history 
    required_default_history = recovery_curve.shape[1]
    default_forecasts = np.pad(default_forecasts, ((0, 0), (0, 0), (required_default_history, 0)), mode="constant")

    # for the loss curve, assume that anything we don't recover by the final month
    # of the recovery curve is recognized as lost
    loss_curve = np.zeros_like(recovery_curve)
    loss_curve[0, -1] = 1 - recovery_curve.sum()
    
    # the natural way to represent the recovery/loss curve is curve[1, i] is the recovery
    # or loss i months after default - however we want to apply these backward looking at
    # time T, so we need to revere them
    recovery_curve = recovery_curve[:, ::-1]
    loss_curve = loss_curve[:, ::-1]
    
    expected_performing_balance = loans[:, 0].copy()
    expected_default_balance = np.zeros_like(expected_performing_balance)

    for i in range(n_months):

        # do full time step projecting cashflows for month i
        scheduled_cashflows, expected_cashflows, expected_default_cashflows = do_full_timestep(
            balance=loans[:, 0],
            expected_performing_balance=expected_performing_balance,
            expected_default_balance=expected_default_balance,
            remaining_term=loans[:, 1],
            time_past_reversion=loans[:, 2],
            fixed_pre_reversion_rate=loans[:, 3],
            post_reversion_margin=loans[:, 4],
            is_interest_only=loans[:, 5],
            cpr=cpr[:, i],
            cdr=cdr[:, i],
            recovery_curve=recovery_curve,
            loss_curve=loss_curve,
            base_rate=base_rate[i],
            default_forecasts=default_forecasts[:, :, i: i + required_default_history]
        )

        # store results of step forward
        cashflow_schedule[:, :, i] = scheduled_cashflows
        cashflow_forecasts[:, :, i] = expected_cashflows
        default_forecasts[:, :, i + required_default_history] = expected_default_cashflows

        # roll forward the inputs for the next step
        loans[:, 0] = scheduled_cashflows[:, -1]  # update the loan balance
        loans[:, 1] -= 1  # decrement loan time remaining by one
        loans[:, 2] += 1  # increment loan time to reversion by one
        expected_performing_balance = expected_cashflows[:, -1]
        expected_default_balance = expected_default_cashflows[:, -1]
    

    # before we return, remove the zero pad from our default forecasts
    default_forecasts = default_forecasts[:, :, required_default_history:]

    return cashflow_schedule, cashflow_forecasts, default_forecasts


def _get_loan_info_at_start_of_forecast(
        forecast_date: dt.date,
        processed_loan_data: pd.DataFrame,
        assumed_mortgage_tenor: int,
        is_interest_only: bool
) -> pd.DataFrame:
    """
    extract key loan information as of the start of our
    forecast period from the processed loan data
    """

    # recover static data from our processed loan data
    static_cols = [
        "loan_id", 
        "product", 
        "pre_reversion_fixed_rate", 
        "post_reversion_boe_margin", 
        "reversion_date"
    ]
    forecast_start_loan_data = processed_loan_data[static_cols].drop_duplicates()
    forecast_start_loan_data["is_interest_only"] = 1 if is_interest_only else 0

    # add the initial outstanding balances and times to reversion at forecast date
    is_forecast_date = processed_loan_data["date"] == forecast_date
    outstanding_balances = processed_loan_data.loc[is_forecast_date, ["loan_id", "month_end_balance"]]
    time_to_reversion = processed_loan_data.loc[is_forecast_date, ["loan_id", "time_to_reversion"]]
    seasoning = processed_loan_data.loc[is_forecast_date, ["loan_id", "seasoning"]]
    forecast_start_loan_data = pd.merge(forecast_start_loan_data, outstanding_balances, on="loan_id")
    forecast_start_loan_data = pd.merge(forecast_start_loan_data, time_to_reversion, on="loan_id")
    forecast_start_loan_data = pd.merge(forecast_start_loan_data, seasoning, on="loan_id")

    # TODO: would be much better to calculate this on a per-mortgage basis rather than
    # making the unrealistic assumption all mortgages are the same length as the one
    # in the s/s example (ie 15 years)
    forecast_start_loan_data["remaining_term"] = assumed_mortgage_tenor * 12 - forecast_start_loan_data["seasoning"]
    return forecast_start_loan_data


def _construct_full_curve_dataframe(
        product_1_curve: pd.Series,
        product_2_curve: pd.Series,
        forecast_start_loan_data: pd.DataFrame,
        min_time_to_reversion: int,
        max_time_to_reversion: int,
    ) -> pd.DataFrame:
    """
    convert our singe product 1/2 base curves into
    an m x n dataframe with one row for each of the 
    m mortgages over n months
    """

    # reindex the curves so that they cover the full required period and 
    # forward/backward fill if required
    product_1_curve = product_1_curve.reindex(list(range(min_time_to_reversion, max_time_to_reversion)))
    product_1_curve = product_1_curve.ffill().bfill()
    product_2_curve = product_2_curve.reindex(list(range(min_time_to_reversion, max_time_to_reversion)))
    product_2_curve = product_2_curve.ffill().bfill()

    # create m x n dataframe for curves with a row for each mortgate
    curves_df = pd.DataFrame(index=forecast_start_loan_data.index, columns=product_1_curve.index)

    # Iterate over each row
    for i in range(forecast_start_loan_data.shape[0]):

        # get correct curve for product type
        is_product_1 = forecast_start_loan_data.iloc[i, :]["product"] == 1
        curve = product_1_curve if is_product_1 else product_2_curve
        
        # shift for months to reversion and store
        time_to_reversion = forecast_start_loan_data.iloc[i, :]["time_to_reversion"] 
        curve = curve.shift(min_time_to_reversion - time_to_reversion).ffill()
        curves_df.iloc[i] = curve
    
    return curves_df


def convert_forecast_results_to_df(
        r_cashflows: np.ndarray,
        r_expected: np.ndarray,
        r_defaults: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    convert the numpy arrays that come out 
    of _run_forecast_internal to easier to 
    handle aggregated cashflow dataframes 
    """

    cashflow_schedule_df = r_cashflows.sum(axis=0)
    cashflow_schedule_df = pd.DataFrame(cashflow_schedule_df, index=[        
        "balance", 
        "payment", 
        "scheduled_interest", 
        "total_scheduled_principal", 
        "closing_balance"
    ])
    cashflow_schedule_df.columns = [i + 1 for i in cashflow_schedule_df.columns]

    expected_cashflows_df = r_expected.sum(axis=0)
    expected_cashflows_df = pd.DataFrame(expected_cashflows_df, index=[
        "expected_performing_balance", 
        "defaults",
        "expected_balance_post_defaults",
        "expected_payment",
        "expected_interest_payment",
        "expected_principal_payment",
        "expected_balance_pre_period_repays",
        "expected_prepayments",
        "expected_closing_balance",    
    ])
    expected_cashflows_df.columns = [i + 1 for i in expected_cashflows_df.columns]

    expected_defaults_df = r_defaults.sum(axis=0)
    expected_defaults_df = pd.DataFrame(expected_defaults_df, index=[
        "expected_default_balance",
        "expected_new_defaults",
        "expected_recovery",
        "expected_loss",
        "expected_closing_default_balance",
    ])
    expected_defaults_df.columns = [i + 1 for i in expected_defaults_df.columns]

    return cashflow_schedule_df, expected_cashflows_df, expected_defaults_df


def run_forecast(
        processed_loan_data: pd.DataFrame,
        forecast_scenario: ForecastScenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    run forecast using the processed loan data and a 
    given forecast scenario
    """

    start_of_forecast_loan_info = _get_loan_info_at_start_of_forecast(
        forecast_date=forecast_scenario.forecast_date,
        processed_loan_data=processed_loan_data,
        assumed_mortgage_tenor=forecast_scenario.assumed_mortgage_tenor,
        is_interest_only=forecast_scenario.interest_only_mortgages
    )

    max_time_to_reversion = start_of_forecast_loan_info["time_to_reversion"].max() + forecast_scenario.forecast_months

    cpr_curve_vector = _construct_full_curve_dataframe(
        product_1_curve=forecast_scenario.product_1_smm,
        product_2_curve=forecast_scenario.product_2_smm,
        forecast_start_loan_data=start_of_forecast_loan_info,
        min_time_to_reversion=-24,  # TODO: would make sense to have this configurable
        max_time_to_reversion=max_time_to_reversion,
    ).values

    cdr_curve_vector = _construct_full_curve_dataframe(
        product_1_curve=forecast_scenario.product_1_mdr,
        product_2_curve=forecast_scenario.product_2_mdr,
        forecast_start_loan_data=start_of_forecast_loan_info,
        min_time_to_reversion=-24,   # TODO: would make sense to have this configurable
        max_time_to_reversion=max_time_to_reversion,
    ).values

    base_rate_vector = np.array([forecast_scenario.boe_base_rate] * cdr_curve_vector.shape[1])
    recovery_curve_vector = forecast_scenario.recovery_curve.values.reshape((1, -1))

    loan_cols = [
        "month_end_balance", 
        "remaining_term", 
        "time_to_reversion", 
        "pre_reversion_fixed_rate", 
        "post_reversion_boe_margin", 
        "is_interest_only"
    ]
    start_of_forecast_loan_info_vector = start_of_forecast_loan_info.loc[:, loan_cols].values

    results = _run_forecast_internal(
        n_months=forecast_scenario.forecast_months,
        loans=start_of_forecast_loan_info_vector,
        cpr=cpr_curve_vector,
        cdr=cdr_curve_vector,
        recovery_curve=recovery_curve_vector,
        base_rate=base_rate_vector
    )

    return results


def _run_forecast_helper(args):
    """
    helper function for multiprocessing so I can 
    provide a single arg
    """
    return run_forecast(*args)


def run_all_forecasts(
        processed_loan_data: pd.DataFrame,
        all_forecasts: list[ForecastScenario]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    run a set of forecasts in parallel and return 
    the combined results
    """
    args = [(processed_loan_data, forecast) for forecast in all_forecasts]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_run_forecast_helper, args)
    return results


# if __name__ == "__main__":

#     from timeit import default_timer as timer

#     import pandas as pd

#     multiplier = 1  # check it works with multiple loans

#     loans = np.array([[100_000, 178, -21, 3.94 / 100, 4.94 / 100, 1]] * multiplier)
#     cpr = np.array([[0.02] * 21 + [0.1] * 179] * multiplier)
#     cdr = np.array([[0.02] * 9 + [0.01] * 191] * multiplier)
#     recovery_curve = np.array([[0, 0, 0, 0, 0, 0.8]])
#     base_rate = np.array([0.045] * 200)

#     t_0 = timer()
    
#     results = _run_forecast_internal(
#         n_months=200,
#         loans=loans,
#         cpr=cpr,
#         cdr=cdr,
#         recovery_curve=recovery_curve,
#         base_rate=base_rate,
#     )
    

#     CASHFLOWS, EXPECTED, DEFAULTS = convert_forecast_results_to_df(*results)

#     t_1 = timer()
#     print(f"Did cashflow forecast for {loans.shape[0]} loans over 200 months in {t_1 - t_0} seconds")
