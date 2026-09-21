import sys
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))


class DashboardTest(unittest.TestCase):
    def test_pages_and_retention_controls(self):
        app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        app.sidebar.selectbox[0].select("🎯 联系优先级").run()
        self.assertFalse(app.exception)
        summary = app.dataframe[0].value
        self.assertEqual(summary["联系人数"].tolist(), [500, 500, 500])
        print("DEFAULT STRATEGY COMPARISON\n", summary.to_string(index=False))
        app.number_input[0].set_value(0).run()
        self.assertFalse(app.exception)
        self.assertTrue((app.dataframe[0].value["联系人数"] == 0).all())
        app.number_input[0].set_value(10000).run()
        self.assertFalse(app.exception)
        self.assertTrue((app.dataframe[0].value["联系人数"] == 1407).all())
        app.sidebar.slider[1].set_range(18.0, 18.0).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("没有测试客户" in w.value for w in app.warning))
        app.sidebar.slider[1].set_range(18.0, 119.0).run()
        for page in ["📈 Insights", "🤖 Predict Churn", "ℹ️ About"]:
            app.sidebar.selectbox[0].select(page).run()
            self.assertFalse(app.exception)
            if page == "🤖 Predict Churn":
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.success or app.error)
