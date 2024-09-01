import random

import pandas as pd
from matplotlib import pyplot as plt


def basic_data_checks(data: pd.DataFrame, name: str, expected_loans: set[str]) -> None:
    """
    plot basic sanity checking on the given data
    """

    global_min = data.min().min()
    global_max = data.max().max()
    missing_loans = expected_loans - set(data.index)
    additional_loans = set(data.index) - expected_loans
    duplicated_loans = data.index[data.index.duplicated()]

    print("=" * 50)
    print(f"SANITY CHECKING FOR {name}")
    print(f"Global min value: {global_min}")
    print(f"Global max value: {global_max}")
    print(f"Missing loans: {missing_loans}")
    print(f"Additional loans: {additional_loans}")
    print(f"Duplicated loans: {duplicated_loans}")

    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(20, 10)

    (data.isnull().sum() / data.shape[0]).plot(kind="line", ax=axs[0, 0], title="Null values (%)")
    data.min().plot(kind="line", ax=axs[0, 1], label="Min value", title="Min and max values")
    data.max().plot(kind="line", ax=axs[0, 1], label="Max value")
    axs[0, 1].legend()

    for i in range(4):
        row = 1 + (i // 2)
        col = i % 2
        n = random.randint(0, len(data) - 1)
        data.iloc[n, :].plot(kind="line", ax=axs[row, col], title=f"Plot for loan {n}")

    fig.tight_layout()
        


def plot_forecast_results(
        cashflow_schedule: pd.DataFrame,
        expected_cashflows: pd.DataFrame,
        expected_defaults: pd.DataFrame
) -> None:

    fig, axs = plt.subplots(5, 2)

    cashflow_schedule.loc["closing_balance", :].round(2).plot(kind="line", ax=axs[0, 0], title="Cashflow schedule - closing balance")
    expected_cashflows.loc["expected_closing_balance", :].round(2).plot(kind="line", ax=axs[0, 1], title="Expected cashflows - closing balance")
    cashflow_schedule.loc["scheduled_interest", :].round(2).plot(kind="line", ax=axs[1, 0], title="Cashflow schedule - interest payments")
    expected_cashflows.loc["expected_interest_payment", :].round(2).plot(kind="line", ax=axs[1, 1], title="Expected cashflows - interest payments")
    cashflow_schedule.loc["total_scheduled_principal", :].round(2).plot(kind="line", ax=axs[2, 0], title="Cashflow schedule - principal payments")
    expected_cashflows.loc["expected_principal_payment", :].round(2).plot(kind="line", ax=axs[2, 1], title="Expected cashflows - principal payments")
    expected_cashflows.loc["defaults", :].round(2).plot(kind="line", ax=axs[3, 0], title="Expected defaults")
    expected_cashflows.loc["expected_prepayments", :].round(2).plot(kind="line", ax=axs[3, 1], title="Expected prepayments")
    expected_defaults.loc["expected_loss", :].round(2).plot(kind="line", ax=axs[4, 0], title="Expected loss")
    expected_defaults.loc["expected_recovery", :].round(2).plot(kind="line", ax=axs[4, 1], title="Expected recovery")

    fig.set_tight_layout(True)
    fig.set_size_inches(25, 20)
