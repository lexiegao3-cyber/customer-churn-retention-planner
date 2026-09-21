# Customer Churn Prediction and Retention Prioritization

A local demonstration built with scikit-learn's Random Forest classifier and Streamlit. The retention planner explores a practical question: **If a customer service team can contact only 500 customers per month, who should they prioritize?**

Compare three outreach strategies, adjust capacity and economic assumptions, and export customer rankings and cost sensitivity results.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m streamlit run dashboard/app.py
```

On macOS, once the environment is installed, you can also double-click `Start Churn Dashboard.command`.

In the sidebar's **Navigate** menu, select **🎯 联系优先级** (Contact Prioritization). The new planner's interface is currently in Chinese.

Locally verified with Python 3.14. Exact dependency versions are recorded in [requirements-local.lock.txt](requirements-local.lock.txt).

## Features and Methodology

- Compare random outreach (fixed seed of 42), highest churn risk first, and a combined risk-and-value strategy.
- Evaluate strategies exclusively on the 20% holdout set from a fixed, stratified 80/20 train/test split. The model is not trained on these customers. Sidebar filters narrow the candidate pool.
- Adjust the contact limit, value horizon, gross margin, assumed retention success rate, and cost per contact.
- Report contact count, actual churners reached, actual churn coverage, churn rate within the selected list, simulated retained customers, total contact cost, and simulated net benefit.
- Export priority lists and cost sensitivity comparisons as CSV files, with the simulation assumptions included.
- Use actual churn labels only for retrospective evaluation, never as model inputs or ranking criteria.

### Economic Assumptions

```text
Customer value proxy = Monthly charge × Value horizon in months × Gross margin

Simulated net benefit per customer =
    Churn probability × Assumed retention success rate × Customer value proxy
    − Contact cost
```

The retention success rate represents the assumed fraction of contacted customers who would otherwise churn but are retained through the intervention.

The combined strategy ranks customers by simulated net benefit. Each strategy selects `min(contact limit, candidate count)` customers, even when simulated net benefit is negative. This provides an equal-capacity comparison; it is not a recommendation to contact customers when doing so is unprofitable.

Contact cost and the other economic assumptions are uniform across customers. Changing a uniform contact cost changes simulated benefit but does not change the ranking. The sensitivity chart illustrates this effect. Customer-specific costs and intervention effects require additional data.

Actual churn coverage is the number of actual churners in the selected list divided by all actual churners in the filtered candidate pool. It is left undefined when the pool contains no actual churners.

## Limitations

High churn risk does not imply that a customer is easy to retain. The retention success rate is a user-specified assumption, unsupported by intervention experiments. Random Forest probabilities have not been calibrated, and simulated benefits are not realized returns. The customer value proxy is not a complete lifetime value estimate.

The dataset is a historical snapshot, so this demonstration cannot establish performance for future months. Repeated inspection of the test set requires subsequent validation on fresh independent data or a later time period.

The original single-customer prediction page exposes only some input fields. Unspecified service fields retain values from the first dataset row. That page remains a demonstration and is not suitable for direct operational use.

The original notebook is retained for reference; its analysis and results have not been revised to reflect these changes. The corrected application and new functionality are in `dashboard/`.

## Changes from the Original Project

- Removed the `Churn` label from model features to fix target leakage.
- Added support for pandas string dtypes and updated Streamlit width parameters.
- Added an independent strategy calculation module, a Chinese-language retention planner, and customer list exports.
- Added a macOS launcher, a dependency lock file, and tests for strategy calculations and dashboard interactions.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover ranking behavior, economic calculations, label-independent selection, capacity limits, empty candidate pools, zero assumed retention success, uniform cost changes, and dashboard interactions.

## Attribution and License

Adapted from [Ayushman Das's Telco Customer Churn Analytics and Prediction](https://github.com/ayushmandas29/Telco-Customer-Churn-Analytics-and-Prediction).

The original Git history and [MIT license](LICENSE) are preserved. The upstream README is retained in [README_UPSTREAM.md](README_UPSTREAM.md); its live demo link points to the original author's application, not this extended version. The bundled cleaned dataset and original notebook are also inherited from the upstream project.
