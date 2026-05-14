from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimizer import optimize
from walk_forward import walk_forward_validation, print_report


def run_basic_optimization(df: pd.DataFrame) -> None:
    """Run standard optimization."""

    print("=" * 72)
    print("STANDARD OPTIMIZATION")
    print("=" * 72)

    results = optimize(
        trades_df=df,
        stop_losses=[5, 10, 15, 20],
        take_profits=[5, 10, 15, 20],
        top_n=5,
    )

    for idx, row in enumerate(results, start=1):
        print(f"\nRank #{idx}")

        for key, value in row.items():
            print(f"{key:<15}: {value}")


def main() -> None:

    data_path = Path("data/trades.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        parse_dates=["entry_time", "exit_time"],
    )

    # Run core optimizer
    run_basic_optimization(df)

    print("\n")

    # Run walk-forward validation
    wf_results = walk_forward_validation(
        trades_df=df,
        stop_losses=[5, 10, 15, 20],
        take_profits=[5, 10, 15, 20],
        train_ratio=0.7,
    )

    print_report(wf_results)

    print("\n")

    # Run benchmark
    from benchmark import run_benchmark
    run_benchmark()

if __name__ == "__main__":
    main()