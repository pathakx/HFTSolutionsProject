# NOTES

## Approach

I implemented a brute-force grid search over all stop-loss (SL) and take-profit (TP) parameter combinations.

For each `(SL, TP)` pair:
- Trades hitting the stop-loss condition are adjusted to `-SL`
- Remaining trades hitting the take-profit condition are adjusted to `+TP`
- All other trades retain their original pnl

The adjusted pnl series is then used to compute:
- Sharpe ratio
- Total pnl
- Number of stopped-out trades
- Number of take-profit trades

The results are sorted deterministically according to the specification:
1. Sharpe descending
2. Total pnl descending
3. `(stop_loss, take_profit)` ascending

---

## Performance Considerations

To satisfy the runtime requirements, I used NumPy vectorized operations instead of row-wise iteration (`iterrows` / `apply`).

The trade columns (`pnl`, `mae`, `mfe`) are converted into NumPy arrays, and boolean masks are used to efficiently apply SL/TP rules across all trades for each parameter combination.

This keeps the implementation both readable and performant for larger parameter sweeps.

---

## Additional Exploration

I also explored additional internal strategy evaluation metrics such as:
- Win rate
- Average win/loss
- Maximum drawdown

Although these metrics were not part of the required output schema, they are commonly used in trading system evaluation and provide additional insight into risk and strategy stability beyond Sharpe ratio alone.

I also created a simple benchmarking script to evaluate runtime performance on larger synthetic datasets and parameter grids.

This helped validate that the vectorized NumPy implementation performs efficiently under the expected workload size.

---

## Edge Cases Handled

The implementation handles:
- Empty input DataFrames
- Empty parameter lists
- Rows containing NaN values
- Single-trade Sharpe calculation
- Zero standard deviation cases
- Deterministic sorting requirements
- Stop-loss priority over take-profit
- Invalid stop-loss / take-profit parameter validation

---

## Tradeoffs

I prioritized correctness and readability first while still ensuring acceptable performance.

The implementation still uses explicit loops over parameter combinations, but all trade-level calculations are vectorized using NumPy.

A fully broadcasted solution across the entire parameter grid may be faster, but would significantly reduce readability and increase implementation complexity for this assignment.

---

## Additional Testing

In addition to the provided public tests, I added extra tests covering:
- Single-trade Sharpe edge cases
- NaN row handling
- Invalid parameter validation

---

## What I Would Improve With More Time

If given more time, I would:
- Explore fully broadcasted grid computation for additional speed improvements
- Add optional profiling/logging utilities
- Implement additional robustness metrics such as profit factor and volatility scaling
- Explore walk-forward or out-of-sample evaluation to reduce overfitting risk