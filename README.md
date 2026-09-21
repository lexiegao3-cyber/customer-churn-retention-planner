# 客户流失预警与联系优先级分析

基于 scikit-learn 随机森林和 Streamlit 的本地演示项目。新增「每月只能联系 500 人」策略比较，可调整名额、价值月数、毛利率、挽留成功率和联系成本，导出名单及成本敏感性结果。

## 本地启动

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.lock.txt
.venv/bin/python -m streamlit run dashboard/app.py
```

已安装环境的 macOS 用户也可以双击 `Start Churn Dashboard.command`。打开网页后，在左侧 Navigate 中选择「🎯 联系优先级」。本地验证环境：Python 3.14；锁定依赖见 requirements-local.lock.txt。

## 新增功能与方法

- 比较随机联系（固定种子 42）、流失风险优先、风险与客户价值综合排序。
- 仅使用固定 80/20 分层划分中的 20% 测试客户做策略回测，模型没有在这些客户上训练；左侧筛选只缩小候选池。
- 输出联系人数、覆盖实际流失人数、实际流失覆盖率、名单流失率、模拟挽留人数、联系成本和模拟净收益。
- 价值代理 = 月费 × 保留月数 × 毛利率。
- 单人模拟净收益 = 流失概率 × 假设挽留成功率 × 价值代理 − 联系成本。
- 综合策略按单人模拟净收益排序。所有策略取 min(名额, 候选人数)，包括净收益为负的情况；这是相同名额的比较，不是建议在负收益下仍联系。
- 联系成本等假设对所有客户相同，因此改变统一成本不会改变排序；敏感性图展示收益变化。客户级成本与干预效果需要额外数据。
- 导出 CSV 包含模拟假设；实际标签仅用于回测，不参与模型输入和排序。

## 重要局限

高流失风险不等于容易挽留。成功率是用户假设，没有干预实验支持；随机森林概率尚未校准，模拟收益不是已实现收益。价值估计不是完整 LTV。数据是历史截面数据，不能证明未来月份的效果。重复查看测试集后，需要新的独立数据或时间外验证。

保留的原始单客户预测页面只填写部分输入，未填写的服务字段沿用第一行样本；该页仍是示例，不宜直接用于实际运营。原始 Notebook 保留作参考，未同步重做其分析或结果；修正及新功能位于 dashboard/。

## 本次修改

- 修复将 Churn 标签作为特征造成的目标泄漏。
- 兼容 pandas 字符串类型，更新 Streamlit 宽度参数。
- 新增独立策略计算模块、中文交互页面及名单导出。
- 添加本地启动入口、依赖锁定、策略与页面验证。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 来源与许可

本项目改编自 [Ayushman Das 的 Telco Customer Churn Analytics and Prediction](https://github.com/ayushmandas29/Telco-Customer-Churn-Analytics-and-Prediction)。保留原 Git 历史和 MIT LICENSE。原作者说明存于 README_UPSTREAM.md，其中在线演示链接属于原作者，不是本次新增版本。仓库自带的清洗数据与原始 Notebook 来自上游。
