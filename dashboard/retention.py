"""Historical holdout ranking and explicitly hypothetical retention economics."""
import pandas as pd

STRATEGIES = ["随机联系", "流失风险优先", "风险 × 客户价值 − 联系成本"]


def compare_strategies(candidates, capacity=500, months=6, margin=0.5,
                       save_rate=0.2, contact_cost=5.0):
    """Labels are used only for evaluation, never for choosing customers.

    Value is a gross-profit proxy, not estimated lifetime value. Save rate is
    an assumed fraction of otherwise-churning contacted customers retained.
    All strategies fill min(capacity, pool size) slots for a fair comparison.
    """
    if capacity < 0 or months < 0 or contact_cost < 0:
        raise ValueError("Capacity, months and cost must be nonnegative")
    if not 0 <= margin <= 1 or not 0 <= save_rate <= 1:
        raise ValueError("Margin and save rate must lie in [0, 1]")
    pool = candidates.copy()
    pool["价值估计 ($)"] = pool["MonthlyCharges"] * months * margin
    pool["模拟净收益 ($)"] = (
        pool["risk"] * save_rate * pool["价值估计 ($)"] - contact_cost
    )
    # Stable ties and a fixed random sample make slider comparisons reproducible.
    pool = pool.sort_values("customerID", kind="stable")
    count = min(int(capacity), len(pool))
    ranked = {
        STRATEGIES[0]: pool.sample(frac=1, random_state=42),
        STRATEGIES[1]: pool.sort_values("risk", ascending=False, kind="stable"),
        STRATEGIES[2]: pool.sort_values("模拟净收益 ($)", ascending=False, kind="stable"),
    }
    churners = int(pool["ChurnLabel"].sum())
    rows, selections = [], {}
    for strategy, ranking in ranked.items():
        selected = ranking.head(count).copy()
        selected.insert(0, "优先级", range(1, len(selected) + 1))
        selections[strategy] = selected
        hits = int(selected["ChurnLabel"].sum())
        rows.append({
            "策略": strategy, "联系人数": len(selected),
            "覆盖实际流失人数": hits,
            "实际流失覆盖率 (%)": 100 * hits / churners if churners else None,
            "名单流失率 (%)": 100 * hits / len(selected) if len(selected) else None,
            "模拟挽留人数": float(selected["risk"].sum() * save_rate),
            "联系总成本 ($)": len(selected) * contact_cost,
            "模拟净收益 ($)": float(selected["模拟净收益 ($)"].sum()),
        })
    return pd.DataFrame(rows), selections
