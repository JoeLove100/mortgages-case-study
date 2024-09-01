import pandas as pd


def _get_current_balance(data: pd.DataFrame) -> pd.Series:
    """
    add start of month outstanding balance
    """

    return data["month_end_balance"].shift(1)


def _get_seasoning(data: pd.DataFrame) -> pd.Series:
    """
    get seasoning (ie months since origination), which
    we take as null pre-origination
    """

    years_between = data["date"].dt.year - data["origination_date"].dt.year
    months_between = data["date"].dt.month - data["origination_date"].dt.month
    seasoning = 12 * years_between + months_between
    return seasoning.where(seasoning >= 0, None)


def _get_consecutive_missed_payments(data: pd.DataFrame) -> pd.Series:
    """
    get the number of consecutive missed payments, which resets
    to zero each time a payment is successfully made
    """

    payment_is_due = data["payment_due"].notnull() & data["payment_due"] != 0
    payment_made_is_zero = data["payment_made"] == 0
    payment_is_missed = payment_is_due & payment_made_is_zero
    cumulative_missed_payments = payment_is_missed.cumsum()
    
    # bit of a trick - create a series where the increments are removed and 
    # forward filled then subtract from the cumulative series to reset to 0
    # on a successful payment
    increments_removed_and_filled = cumulative_missed_payments.where(~payment_is_missed).ffill()
    return cumulative_missed_payments - increments_removed_and_filled


def _get_prepay_flag(data: pd.DataFrame) -> pd.Series:
    """
    get series where we have a boolean flag for the 
    month in which pre-payment occurs
    """

    this_month_balance_zero = data["month_end_balance"] == 0
    prev_month_balance_not_zero = data["month_end_balance"].shift(1) > 0
    return this_month_balance_zero & prev_month_balance_not_zero


def _get_default_flag(data: pd.DataFrame) -> pd.Series:
    """
    get series in which we have a boolean flag for the
    month in which a default (>= 3 missed payments in a row)
    occurs
    """

    is_defaulted_this_month = data["n_missed_payments"].cummax() >= 3
    not_defaulted_so_far = ~(is_defaulted_this_month.shift(1).fillna(False))
    return is_defaulted_this_month & not_defaulted_so_far


def _get_is_recovery_payment_flag(data: pd.DataFrame) -> pd.Series:
    """
    boolean flag for each payment showing whether it 
    was made after default or not
    """

    has_defaulted = data["n_missed_payments"].cummax() >= 3
    payment_made = data["payment_made"].notnull() & data["payment_made"] > 0
    return has_defaulted & payment_made


def _get_recovery_flag(data: pd.DataFrame) -> pd.Series:
    """
    get series in which we have boolean flag for the
    month in which the first recovery payment occurs
    """

    n_recovery_payments_so_far = data["is_recovery_payment"].cumsum()
    no_previous_recovery_payments = n_recovery_payments_so_far.shift(1) == 0
    return data["is_recovery_payment"] & no_previous_recovery_payments 


def get_time_to_reversion(data: pd.DataFrame) -> pd.Series:
    """
    get series showing time to reversion, which is None
    after the reversion
    """

    years = data["date"].dt.year - data["reversion_date"].dt.year
    months = data["date"].dt.month - data["reversion_date"].dt.month
    time_to_reversion = 12 * years + months
    return time_to_reversion


def _get_is_post_seller_purchase_date(data: pd.DataFrame) -> pd.Series:
    """
    get boolean series of whether or not this is on or
    after the month the seller purchased the loan in
    """

    return data["date"] >= data["investor_1_acquisition_date"]


def _get_total_post_default_recovery_payments(data: pd.DataFrame) -> float | None:
    """
    get the total post default recovery payments - will return None if
    no default occurred
    """

    filtered_data = data.loc[data["is_recovery_payment"], "payment_made"]
    if filtered_data.empty:
        return None
    return filtered_data.sum()


def _get_prepayment_date(data: pd.DataFrame) -> pd.Timestamp | None:
    """
    get the prepayment date, or return None if no prepayment
    """

    filtered_data = data.loc[data["prepaid_in_month"], "date"]
    if filtered_data.empty:
        return None
    return filtered_data.iloc[0]


def _get_default_date(data: pd.DataFrame) -> pd.Timestamp | None:
    """
    get the default date, or return None if no default
    """

    filtered_data = data.loc[data["default_in_month"], "date"]
    if filtered_data.empty:
        return None
    return filtered_data.iloc[0]


def _get_recovery_date(data: pd.DataFrame) -> pd.Timestamp | None:
    """
    get the recovery date, or return None if no recovery
    """

    filtered_data = data.loc[data["recovery_in_month"], "date"]
    if filtered_data.empty:
        return None
    return filtered_data.iloc[0]


def _get_exposure_at_default(data: pd.DataFrame) -> float | None:
    """
    get current loan balance outstanding in the month of
    default, or return None if no default occurred
    """

    filtered_data = data.loc[data["default_in_month"], "current_balance"]
    if filtered_data.empty:
        return None
    return filtered_data.iloc[0]


def add_dynamic_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    add a series of standard dynamic columns we will
    need for our modelling 
    """

    data["current_balance"] = _get_current_balance(data)
    data["seasoning"] = _get_seasoning(data)
    data["n_missed_payments"] = _get_consecutive_missed_payments(data)
    data["prepaid_in_month"] = _get_prepay_flag(data)
    data["default_in_month"] = _get_default_flag(data)
    data["is_recovery_payment"] = _get_is_recovery_payment_flag(data)
    data["recovery_in_month"] = _get_recovery_flag(data)
    data["time_to_reversion"] = get_time_to_reversion(data)
    data["is_post_seller_purchase_date"] = _get_is_post_seller_purchase_date(data)

    data["has_defaulted"] = data["n_missed_payments"].cummax() >= 3
    data["has_prepaid"] = data["current_balance"] == 0
    data["has_recovered"] = data["recovery_in_month"].cumsum()
    
    return data


def add_static_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    add static columns (ie those that do not change each month)
    to the data
    """

    data["postdefault_recoveries"] = _get_total_post_default_recovery_payments(data)
    data["prepayment_date"] = _get_prepayment_date(data)
    data["date_of_default"] = _get_default_date(data)
    data["date_of_recovery"] = _get_recovery_date(data)
    data["exposure_at_default"] = _get_exposure_at_default(data)
    data["recovery_percent"] = data["postdefault_recoveries"] / data["exposure_at_default"]

    return data
