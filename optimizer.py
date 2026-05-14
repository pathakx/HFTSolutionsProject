from __future__ import annotations

import numpy as np
import pandas as pd


def optimize(
    trades_df: pd.DataFrame,
    stop_losses: list[float],
    take_profits: list[float],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N (stop_loss, take_profit) combinations by Sharpe."""

    # Edge cases
    if trades_df.empty:
        return []

    if not stop_losses or not take_profits:
        return []

    # Validate parameter values
    if any(sl <= 0 for sl in stop_losses):
        raise ValueError("All stop_losses must be positive.")

    if any(tp <= 0 for tp in take_profits):
        raise ValueError("All take_profits must be positive.")

    # Drop rows with NaNs in required columns
    df = trades_df.dropna(subset=["pnl", "mae", "mfe"])

    if df.empty:
        return []

    # Convert to NumPy arrays for performance
    pnl = df["pnl"].to_numpy(dtype=float)
    mae = df["mae"].to_numpy(dtype=float)
    mfe = df["mfe"].to_numpy(dtype=float)

    results = []

    # Grid search over all SL/TP combinations
    for sl in stop_losses:
        for tp in take_profits:

            # SL has priority
            sl_mask = mae >= sl

            # TP only triggers if SL did not trigger
            tp_mask = (~sl_mask) & (mfe >= tp)

            # Adjusted pnl
            adjusted = pnl.copy()

            adjusted[sl_mask] = -sl
            adjusted[tp_mask] = tp

            # Additional internal metrics (not returned)

            wins = adjusted[adjusted > 0]
            losses = adjusted[adjusted < 0]

            win_rate = (
                float(len(wins) / len(adjusted))
                if len(adjusted) > 0
                else 0.0
            )

            avg_win = (
                float(wins.mean())
                if len(wins) > 0
                else 0.0
            )

            avg_loss = (
                float(losses.mean())
                if len(losses) > 0
                else 0.0
            )

            # Max drawdown
            cumulative = np.cumsum(adjusted)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative

            max_drawdown = (
                float(drawdowns.max())
                if len(drawdowns) > 0
                else 0.0
            )

            # Required metrics
            total_pnl = float(adjusted.sum())

            stopped_out = int(sl_mask.sum())
            took_profit = int(tp_mask.sum())

            # Sharpe ratio
            if len(adjusted) <= 1:
                sharpe = 0.0
            else:
                std = adjusted.std(ddof=0)

                if std == 0 or np.isnan(std):
                    sharpe = 0.0
                else:
                    sharpe = float(adjusted.mean() / std)

            results.append(
                {
                    "stop_loss": float(sl),
                    "take_profit": float(tp),
                    "sharpe": float(sharpe),
                    "total_pnl": float(total_pnl),
                    "stopped_out": stopped_out,
                    "took_profit": took_profit,
                }
            )

    # Deterministic sorting
    results.sort(
        key=lambda x: (
            -x["sharpe"],
            -x["total_pnl"],
            x["stop_loss"],
            x["take_profit"],
        )
    )

    return results[:top_n]