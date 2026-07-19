# egs 使用说明

本目录提供 AUC 指标的本地计算示例，帮助选手理解评价方式。请注意：这里的本地示例格式和线上正式提交格式不同。

## 1. 线上正式提交格式

线上提交只需要一个 CSV 文件，编码为 UTF-8，第一行为表头：

```csv
id,posterior
seen_pair_000001,0.5
seen_pair_000002,0.5
unseen_pair_000001,0.5
unseen_pair_000002,0.5
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | str | 测试样本编号。seen 子集使用 `seen_pair_000001` 格式，unseen 子集使用 `unseen_pair_000001` 格式。 |
| posterior | float | 模型输出的唤醒后验概率，取值范围为 `[0, 1]`，数值越大表示越倾向于唤醒。 |

正式提交不需要也不得包含 `label` 字段。平台会使用隐藏测试集标签分别计算 seen / unseen 两个子集的 AUC，并取二者平均作为最终分数。

## 2. 本地示例脚本

`cal_auc.py` 是本地 AUC 计算示例脚本。由于本地计算 AUC 需要真实标签，因此示例文件 `example.csv` 包含 `label` 字段：

```csv
id,posterior,label
000001,0.639427,0
000002,0.74155,0
```

运行方式：

```bash
python cal_auc.py --csv example.csv
```

该脚本仅用于本地理解指标或在有标签数据上自测，不代表线上正式提交格式。

## 3. 重要提醒

- 线上提交时请将 `eval_seen` 和 `eval_unseen` 两个测试子集的预测结果合并到同一个 CSV 文件中。
- 线上提交 ID 必须带子集前缀：`seen_` 或 `unseen_`。
- 线上提交文件只包含 `id,posterior` 两列。
- `posterior` 必须是 `[0, 1]` 范围内的数值。
- 不要在正式提交文件中包含 `label` 字段。
