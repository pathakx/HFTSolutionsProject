from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimizer import optimize


def walk_forward_validation(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    train_ratio: float = 0.7,
) -> dict:
    """
    Perform simple out-of-sample validation.

    Workflow:
    1. Sort trades chronologically
    2. Split into train/test periods
    3. Optimize parameters on training data
    4. Evaluate best parameters on unseen test data

    Returns:
        dict containing:
        - best training result
        - out-of-sample test result
        - dataset statistics
    """

    required_columns = {
        "entry_time",
        "exit_time",
        "pnl",
        "mae",
        "mfe",
    }

    missing = required_columns - set(trades_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if trades_df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    # Sort chronologically
    df = trades_df.sort_values("entry_time").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    if split_idx == 0 or split_idx >= len(df):
        raise ValueError(
            "Dataset too small for train/test split."
        )

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Optimize on training data
    train_results = optimize(
        train_df,
        stop_losses=stop_losses,
        take_profits=take_profits,
        top_n=1,
    )

    if not train_results:
        raise ValueError(
            "Optimization produced no training results."
        )

    best_train = train_results[0]

    best_sl = best_train["stop_loss"]
    best_tp = best_train["take_profit"]

    # Evaluate on unseen test data
    test_results = optimize(
        test_df,
        stop_losses=[best_sl],
        take_profits=[best_tp],
        top_n=1,
    )

    if not test_results:
        raise ValueError(
            "Optimization produced no test results."
        )

    best_test = test_results[0]

    return {
        "train_period_trades": int(len(train_df)),
        "test_period_trades": int(len(test_df)),
        "best_train_result": best_train,
        "out_of_sample_result": best_test,
    }


def print_report(results: dict) -> None:
    """Pretty-print walk-forward validation results."""

    print("=" * 72)
    print("WALK-FORWARD VALIDATION REPORT")
    print("=" * 72)

    print("\nDataset Split")
    print("-" * 72)
    print(f"Training trades : {results['train_period_trades']}")
    print(f"Testing trades  : {results['test_period_trades']}")

    train = results["best_train_result"]

    print("\nBest Parameters Found On Training Data")
    print("-" * 72)

    for key, value in train.items():
        print(f"{key:<15}: {value}")

    test = results["out_of_sample_result"]

    print("\nOut-Of-Sample Performance")
    print("-" * 72)

    for key, value in test.items():
        print(f"{key:<15}: {value}")

    print("\nInterpretation")
    print("-" * 72)

    if test["sharpe"] > 0:
        print(
            "The optimized parameters retained positive "
            "risk-adjusted performance on unseen data."
        )
    else:
        print(
            "The optimized parameters did not generalize "
            "well to unseen data."
        )

    print("=" * 72)


if __name__ == "__main__":

    data_path = Path("data/trades.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        parse_dates=["entry_time", "exit_time"],
    )

    results = walk_forward_validation(
        trades_df=df,
        stop_losses=[5, 10, 15, 20],
        take_profits=[5, 10, 15, 20],
        train_ratio=0.7,
    )

    print_report(results)