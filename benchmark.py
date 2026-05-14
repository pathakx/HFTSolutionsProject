import time

import numpy as np
import pandas as pd

from optimizer import optimize

def run_benchmark():
    # Synthetic benchmark dataset
    N = 5000

    df = pd.DataFrame({
        "trade_id": np.arange(N),
        "entry_time": pd.Timestamp("2024-01-01"),
        "exit_time": pd.Timestamp("2024-01-02"),
        "pnl": np.random.normal(0, 10, N),
        "mae": np.random.uniform(0, 20, N),
        "mfe": np.random.uniform(0, 20, N),
    })

    # 32 x 32 parameter grid
    sls = np.linspace(1, 20, 32)
    tps = np.linspace(1, 20, 32)

    print(f"Running benchmark with {N} trades")
    print(f"Parameter combinations: {len(sls) * len(tps)}")

    start = time.perf_counter()

    result = optimize(
        df,
        list(sls),
        list(tps),
    )

    end = time.perf_counter()

    runtime = end - start

    print(f"\nCompleted in {runtime:.4f} seconds")

    best = result[0]

    print("\nBest Result:")
    print(
        f"SL={best['stop_loss']}, "
        f"TP={best['take_profit']}, "
        f"Sharpe={best['sharpe']:.4f}"
    )