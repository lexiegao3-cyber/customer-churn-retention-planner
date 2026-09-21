import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from retention import compare_strategies, STRATEGIES


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.pool = pd.DataFrame({
            "customerID": ["a", "b", "c", "d"],
            "risk": [.9, .7, .2, .1],
            "MonthlyCharges": [10., 100., 100., 10.],
            "ChurnLabel": [1, 0, 1, 0],
        })

    def test_rankings_and_economics(self):
        summary, selected = compare_strategies(self.pool, 1, 1, 1, .5, 5)
        self.assertEqual(selected[STRATEGIES[1]].customerID.tolist(), ["a"])
        self.assertEqual(selected[STRATEGIES[2]].customerID.tolist(), ["b"])
        risk = summary.set_index("策略").loc[STRATEGIES[1]]
        self.assertEqual(risk["实际流失覆盖率 (%)"], 50)
        self.assertAlmostEqual(risk["模拟净收益 ($)"], -.5)
        self.assertAlmostEqual(summary.iloc[2]["模拟净收益 ($)"], 30.)

    def test_labels_cannot_change_selection(self):
        _, first = compare_strategies(self.pool, 2)
        changed = self.pool.assign(ChurnLabel=1-self.pool.ChurnLabel)
        _, second = compare_strategies(changed, 2)
        for name in STRATEGIES:
            self.assertEqual(first[name].customerID.tolist(), second[name].customerID.tolist())

    def test_capacity_empty_and_no_churn(self):
        summary, _ = compare_strategies(self.pool, 500)
        self.assertTrue((summary["联系人数"] == 4).all())
        for pool in [self.pool, self.pool.iloc[:0]]:
            summary, _ = compare_strategies(pool, 0)
            self.assertTrue((summary["联系人数"] == 0).all())
            self.assertTrue((summary["模拟净收益 ($)"] == 0).all())
        summary, _ = compare_strategies(self.pool.assign(ChurnLabel=0))
        self.assertTrue(summary["实际流失覆盖率 (%)"].isna().all())

    def test_uniform_cost_changes_profit_not_selection(self):
        low, a = compare_strategies(self.pool, 2, contact_cost=5)
        high, b = compare_strategies(self.pool, 2, contact_cost=15)
        for name in STRATEGIES:
            self.assertEqual(a[name].customerID.tolist(), b[name].customerID.tolist())
        self.assertTrue(((low["模拟净收益 ($)"]-high["模拟净收益 ($)"]-20).abs() < 1e-9).all())

    def test_zero_success_is_only_cost(self):
        summary, _ = compare_strategies(self.pool, 2, save_rate=0, contact_cost=5)
        self.assertTrue((summary["模拟净收益 ($)"] == -10).all())


if __name__ == "__main__":
    unittest.main()
