import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
from retention import compare_strategies, STRATEGIES

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============ PATHS & CONFIG ============

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
)


# ============ LOAD DATA & TRAIN MODEL (CACHED) ============

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Ensure correct types
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train churn prediction model and return pipeline + feature columns + metrics."""
    target = "ChurnLabel"
    y = df[target]

    feature_cols = [c for c in df.columns if c not in [target, "Churn", "customerID"]]
    X = df[feature_cols]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    holdout = df.loc[X_test.index].copy()
    holdout["risk"] = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return model, feature_cols, numeric_features, categorical_features, metrics, holdout


df = load_data()
model, feature_cols, num_feats, cat_feats, model_metrics, holdout = train_model(df)


# ============ SIDEBAR: THEME & FILTERS & NAV ============

st.sidebar.title("⚙️ Controls")

# Theme toggle (changes chart style)
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
plot_template = "plotly_dark" if theme_choice == "Dark" else "plotly_white"

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "🔍 Navigate",
    ["🏠 Home", "📈 Insights", "🎯 联系优先级", "🤖 Predict Churn", "ℹ️ About"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

# Multi-select filters
gender_filter = st.sidebar.multiselect(
    "Gender", sorted(df["gender"].dropna().unique()), default=sorted(df["gender"].dropna().unique())
)

senior_filter = st.sidebar.multiselect(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    default=[0, 1],
)

contract_filter = st.sidebar.multiselect(
    "Contract Type", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique())
)

internet_filter = st.sidebar.multiselect(
    "Internet Service", sorted(df["InternetService"].dropna().unique()),
    default=sorted(df["InternetService"].dropna().unique())
)

payment_filter = st.sidebar.multiselect(
    "Payment Method", sorted(df["PaymentMethod"].dropna().unique()),
    default=sorted(df["PaymentMethod"].dropna().unique())
)

# Range filters
tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.sidebar.slider("Tenure (months)", tenure_min, tenure_max, (tenure_min, tenure_max))

mc_min, mc_max = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
monthly_range = st.sidebar.slider(
    "Monthly Charges",
    float(np.floor(mc_min)),
    float(np.ceil(mc_max)),
    (float(np.floor(mc_min)), float(np.ceil(mc_max))),
)


def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()

    if gender_filter:
        d = d[d["gender"].isin(gender_filter)]
    if senior_filter:
        d = d[d["SeniorCitizen"].isin(senior_filter)]
    if contract_filter:
        d = d[d["Contract"].isin(contract_filter)]
    if internet_filter:
        d = d[d["InternetService"].isin(internet_filter)]
    if payment_filter:
        d = d[d["PaymentMethod"].isin(payment_filter)]

    d = d[(d["tenure"] >= tenure_range[0]) & (d["tenure"] <= tenure_range[1])]
    d = d[(d["MonthlyCharges"] >= monthly_range[0]) & (d["MonthlyCharges"] <= monthly_range[1])]

    return d


filtered_df = apply_filters(df)


# ============ REUSABLE KPIs ============

def render_kpis(d: pd.DataFrame):
    total_customers = len(d)
    churn_rate = d["ChurnLabel"].mean() * 100 if len(d) > 0 else 0.0
    avg_monthly = d["MonthlyCharges"].mean() if len(d) > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", total_customers)
    c2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    c3.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    c4.metric("Model F1 Score", f"{model_metrics['f1']:.2f}")


# ============ PAGE: HOME ============

if page == "🏠 Home":
    st.title("📊 Telco Customer Churn — Enterprise Analytics Dashboard")

    st.caption("Filters applied on left sidebar. All statistics & visuals below reflect the filtered data.")

    render_kpis(filtered_df)

    st.markdown("---")
    st.subheader("📌 Sample Customer Records")
    st.dataframe(filtered_df.head(20), width="stretch")

    # Download button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="filtered_telco_churn_customers.csv",
        mime="text/csv",
    )


# ============ PAGE: INSIGHTS ============

elif page == "📈 Insights":
    st.title("📈 Churn Insights")

    if filtered_df.empty:
        st.warning("No data to display. Please relax filters in the sidebar.")
    else:
        render_kpis(filtered_df)
        st.markdown("---")

        # Churn rate by contract
        st.subheader("🔹 Churn Rate by Contract Type")
        contract_stats = (
            filtered_df.groupby("Contract")["ChurnLabel"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnLabel": "ChurnRate"})
        )
        fig1 = px.bar(
            contract_stats,
            x="Contract",
            y="ChurnRate",
            labels={"ChurnRate": "Churn Rate"},
            text=(contract_stats["ChurnRate"] * 100).round(1).astype(str) + "%",
            template=plot_template,
        )
        fig1.update_traces(textposition="outside")
        fig1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig1, width="stretch")

        # Churn by tenure
        st.subheader("🔹 Churn by Tenure")
        fig2 = px.histogram(
            filtered_df,
            x="tenure",
            color="Churn",
            barmode="group",
            nbins=40,
            template=plot_template,
        )
        st.plotly_chart(fig2, width="stretch")

        # Monthly charges vs churn
        st.subheader("🔹 Monthly Charges vs Churn")
        fig3 = px.box(
            filtered_df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            template=plot_template,
        )
        st.plotly_chart(fig3, width="stretch")


# ============ PAGE: RETENTION PLANNER ============

elif page == "🎯 联系优先级":
    st.title("🎯 每月只能联系 500 人，优先联系谁？")
    st.write("比较三种联系策略，在有限名额下找到更多高风险客户，并探索不同成本假设下的收益。")
    st.info("历史回测演示：仅使用未参与模型训练的 20% 测试客户。实际流失标签只用于回顾评估；名单不是当前在网客户的真实营销任务。左侧筛选会作用于候选池。")
    candidates = apply_filters(holdout)
    c1, c2, c3 = st.columns(3)
    c1.metric("候选客户", f"{len(candidates):,}")
    c2.metric("候选池实际流失人数", int(candidates["ChurnLabel"].sum()))
    c3.metric("未参与训练的客户总数", len(holdout))

    st.subheader("设置联系名额与模拟假设")
    a, b, c = st.columns(3)
    capacity = a.number_input("每月最多联系人数", min_value=0, max_value=10000, value=500, step=50)
    months = b.slider("保留价值计算月数", 1, 24, 6)
    margin = c.slider("毛利率假设 (%)", 0, 100, 50) / 100
    a, b = st.columns(2)
    save_rate = a.slider("挽留成功率假设 (%)", 0, 100, 20) / 100
    contact_cost = b.number_input("每位客户联系成本 ($)", min_value=0.0, max_value=10000.0, value=5.0, step=1.0)
    st.caption("三种策略均使用相同名额（候选不足时取全部）。随机策略固定种子 42，作为一次可复现的抽样基线。")
    st.warning("高流失风险 ≠ 容易被挽留。成功率是假设，不是模型估计或已验证效果；概率尚未校准。以下收益为情景模拟，不是实际收入或因果结论。")
    with st.expander("计算方法与边界"):
        st.markdown("""
- 客户价值代理 = 月费 × 保留月数 × 毛利率，不等于完整客户终身价值。
- 单人模拟净收益 = 流失概率 × 假设挽留成功率 × 客户价值代理 − 联系成本。
- 风险策略按流失概率排序；综合策略按模拟净收益排序；排序均不使用实际流失标签。
- 实际流失覆盖率 = 名单中实际流失人数 ÷ 筛选后候选池全部实际流失人数。分母为 0 时显示为空。
- 当前每人的联系成本、成功率、价值月数和毛利率相同。正比例参数下，综合排序等价于概率 × 月费；改变统一成本会改变收益，但不会改变排序。
- 所有策略填满给定名额用于公平比较，即使模拟收益为负。真实运营中可以少联系，并需考虑优惠成本、客户级响应差异和实验验证。
- 本数据是历史截面数据，没有按月追踪的未来流失标签；本页无法证明下一月预测效果。反复查看测试结果后，不应把它当作最终独立验证。
""")
    if candidates.empty:
        st.warning("当前筛选下没有测试客户，请放宽左侧筛选条件。")
    else:
        summary, selections = compare_strategies(candidates, capacity, months, margin, save_rate, contact_cost)
        st.subheader("同样名额，三种策略的结果")
        st.dataframe(summary.round(2), hide_index=True, width="stretch")
        fig = px.bar(summary, x="策略", y="实际流失覆盖率 (%)", color="策略", template=plot_template)
        st.plotly_chart(fig, width="stretch")
        st.subheader("联系成本变化，模拟净收益如何变化？")
        sensitivity = []
        for cost in sorted(set([0.0, 5.0, 10.0, 20.0, float(contact_cost)])):
            result, _ = compare_strategies(candidates, capacity, months, margin, save_rate, cost)
            result["每位客户联系成本 ($)"] = cost
            sensitivity.append(result)
        sensitivity = pd.concat(sensitivity, ignore_index=True)
        chart = px.line(sensitivity, x="每位客户联系成本 ($)", y="模拟净收益 ($)", color="策略", markers=True, template=plot_template)
        chart.add_hline(y=0, line_dash="dash")
        st.plotly_chart(chart, width="stretch")
        st.subheader("查看与导出优先联系名单")
        strategy = st.selectbox("名单策略", STRATEGIES, index=1)
        selected = selections[strategy]
        display = selected[["优先级", "customerID", "risk", "MonthlyCharges", "Contract", "tenure", "价值估计 ($)", "模拟净收益 ($)", "ChurnLabel"]].rename(columns={
            "customerID": "客户编号", "risk": "流失概率", "MonthlyCharges": "月费 ($)",
            "Contract": "合同类型", "tenure": "在网月数", "ChurnLabel": "历史实际流失 (仅回测)",
        })
        st.dataframe(display.round(4), hide_index=True, width="stretch")
        # Include assumptions so exported estimates retain their context.
        exported = display.assign(策略=strategy, 价值月数=months, 毛利率假设=margin,
                                  挽留成功率假设=save_rate, 单人联系成本=contact_cost,
                                  用途="历史测试集回测；收益为假设模拟")
        st.download_button("⬇️ 下载联系名单与模拟假设", exported.to_csv(index=False).encode("utf-8-sig"), "retention_priority_list.csv", "text/csv")
        st.download_button("⬇️ 下载成本敏感性比较", sensitivity.assign(价值月数=months, 毛利率假设=margin, 挽留成功率假设=save_rate).to_csv(index=False).encode("utf-8-sig"), "retention_cost_scenarios.csv", "text/csv")


# ============ PAGE: PREDICT CHURN ============

elif page == "🤖 Predict Churn":
    st.title("🤖 Churn Prediction — What-if Analysis")

    st.write("Fill in the customer details below and click **Predict** to see whether the customer is likely to churn.")

    # Use a base row to ensure all columns exist
    base = df.iloc[[0]].copy()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", sorted(df["Partner"].dropna().unique()))

        with col2:
            dependents = st.selectbox("Dependents", sorted(df["Dependents"].dropna().unique()))
            tenure_val = st.slider("Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()), 12)
            contract = st.selectbox("Contract", sorted(df["Contract"].dropna().unique()))

        with col3:
            internet = st.selectbox("Internet Service", sorted(df["InternetService"].dropna().unique()))
            payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].dropna().unique()))
            monthly_val = st.number_input(
                "Monthly Charges",
                float(df["MonthlyCharges"].min()),
                float(df["MonthlyCharges"].max()),
                float(df["MonthlyCharges"].mean()),
            )

        total_val = st.number_input(
            "Total Charges",
            float(df["TotalCharges"].min()),
            float(df["TotalCharges"].max()),
            float(df["TotalCharges"].mean()),
        )

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        # Update base row with user choices
        base["gender"] = gender
        base["SeniorCitizen"] = senior
        base["Partner"] = partner
        base["Dependents"] = dependents
        base["tenure"] = tenure_val
        base["Contract"] = contract
        base["InternetService"] = internet
        base["PaymentMethod"] = payment
        base["MonthlyCharges"] = monthly_val
        base["TotalCharges"] = total_val

        X_input = base[feature_cols]

        pred_proba = model.predict_proba(X_input)[0][1]
        pred_label = model.predict(X_input)[0]

        st.markdown("---")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            if pred_label == 1:
                st.error(f"⚠ The customer is **LIKELY TO CHURN**.\n\nEstimated probability: **{pred_proba:.2%}**")
            else:
                st.success(f"✅ The customer is **NOT LIKELY TO CHURN**.\n\nEstimated probability: **{pred_proba:.2%}**")

        with col_right:
            st.metric("Churn Probability", f"{pred_proba:.2%}")
            st.metric("Model Accuracy", f"{model_metrics['accuracy']:.2f}")
            st.metric("Model F1 Score", f"{model_metrics['f1']:.2f}")


# ============ PAGE: ABOUT ============

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.write(
        """
        This is an **enterprise-style Customer Churn Analytics & Prediction system** built using:

        - **Python, Pandas, NumPy**
        - **Scikit-learn** for machine learning
        - **Streamlit** for the interactive web dashboard
        - **Plotly** for interactive visualizations
        - A real-world **Telco Customer Churn** dataset

        ### Key Capabilities
        - Global filters to slice & dice customer segments
        - KPIs: total customers, churn rate, average revenue
        - Interactive insights on churn vs contract, tenure, and charges
        - Machine learning model (Random Forest) to predict churn probability
        - What-if analysis via simulated customer profiles
        - One-click download of filtered customer data

        This project demonstrates an end-to-end data pipeline from **cleaned dataset → ML model → business dashboard**,
        similar to what is deployed in real analytics teams in industry.
        """
    )

    st.markdown("---")
    st.subheader("📊 Dataset Snapshot")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("📈 Feature Summary (Numeric)")
    st.write(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())
