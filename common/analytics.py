import pandas as pd


def _get_monthly_mortality(data: pd.DataFrame, mortality_col: str, index_col: str) -> pd.Series:
    """
    get monthly `mortality` for the given mortality column (ie default or prepayment
    essentially) conditional on the mortgage being neither defaulted nor prepaid
    at the start of the month
    """

    # get balance @ start of month for months with mortality event (prepay/default)
    mortality_in_month = data.loc[data[mortality_col]]
    mortality_in_month = mortality_in_month.groupby(index_col)["current_balance"].sum()
    
    # get total amount outstanding @ the start of each month
    not_prepaid_or_defaulted = ~(data["has_defaulted"] | data["has_prepaid"])
    outstanding_balance_at_som = data.loc[not_prepaid_or_defaulted]
    outstanding_balance_at_som = outstanding_balance_at_som.groupby(index_col)["current_balance"].sum()

    # take the ratio of the two and return
    smm = (mortality_in_month / outstanding_balance_at_som).fillna(0)
    return smm


def get_smm(data: pd.DataFrame, index_col: str) -> pd.Series:
    """
    return the single monthly mortality of the 
    mortgage pool
    """ 

    return _get_monthly_mortality(data, mortality_col="prepaid_in_month", index_col=index_col)


def get_mdr(data: pd.DataFrame, index_col: str) -> pd.Series:
    """
    return the monthly default risk of the
    mortgage pool
    """

    return _get_monthly_mortality(data, mortality_col="default_in_month", index_col=index_col)


def get_conditional_prepayment_curve(data: pd.DataFrame, index_col: str, pivots: list[int] = None) -> pd.Series | pd.DataFrame:
    """
    get CPR (conditional pre-payment rate) - will return a series with seasoning as the 
    index if no pivots provided, or a dataframe with the pivots as the columns if
    they are provided
    """

    single_monthly_mortality = get_smm(data, index_col)
    cpr = 1 - (1 - single_monthly_mortality) ** 12
    if not pivots:
        return cpr
    
    return cpr.loc[pivots].to_frame()  # convert to pandas dataframe


def get_conditional_default_curve(data: pd.DataFrame, index_col: str, pivots: list[int] = None) -> pd.Series | pd.DataFrame:
    """
    get CDR (conditional default rate) - will return a series with seasoning as the 
    index if no pivots are provided, or a dataframe with the pivots as the columns if
    they are provided
    """

    monthly_default_rate = get_mdr(data, index_col=index_col)
    cdr = 1 - (1 - monthly_default_rate) ** 12
    if not pivots:
        return cdr
    
    return cdr.loc[pivots].to_frame()  # convert to pandas dataframe


def _get_months_since_default(data: pd.DataFrame) -> pd.Series:
    """
    get series giving the number of months since the loan defaulted,
    or None if there is no default/it has not defaulted yet
    """

    years = data["date"].dt.year - data["date_of_default"].dt.year
    months = data["date"].dt.month - data["date_of_default"].dt.month
    months_since_default = 12 * years + months
    return months_since_default.where(months_since_default >= 0, None)

def get_recovery_curve(data: pd.DataFrame) -> pd.Series:
    """
    get cumulative recovery by month since default as a %
    of the total exposure at default
    """

    # get total exposure @ default
    exposure_at_default_by_loan = data[["loan_id", "exposure_at_default"]].drop_duplicates()
    total_exposure_at_default = exposure_at_default_by_loan["exposure_at_default"].sum()

    # get the recovery amt by months since the default occurred
    loans_with_default = data[data["date_of_default"].notnull()].copy()
    loans_with_default["months_since_default"] = _get_months_since_default(loans_with_default)
    recovery_payments = loans_with_default.groupby("months_since_default")["payment_made"].sum()
    cumulative_recovery_payments = recovery_payments.cumsum()

    # return the ratio of the two
    return cumulative_recovery_payments / total_exposure_at_default


def get_monthly_recovery_prob(data: pd.DataFrame) -> pd.Series:
    """
    get monthly probability of a loan recovering based
    on months since default occurred
    """

    # get only the loans which default
    loans_with_default = data[data["date_of_default"].notnull()].copy()
    loans_with_default["months_since_default"] = _get_months_since_default(loans_with_default)

    # now get a count of defaulted loans by months_since_default
    loans_no_recovery = loans_with_default.loc[loans_with_default["has_recovered"] == 0]
    loans_no_recovery = loans_no_recovery.groupby("months_since_default")["loan_id"].count()

    # take the change month-on-month divided by the total at start of month
    prob_of_recovery = (loans_no_recovery.diff(-1) / loans_no_recovery).shift(1)
    return prob_of_recovery
