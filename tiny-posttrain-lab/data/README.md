# 数据目录说明

原则：**不提交大文件、不提交未授权数据。**

## SFT

默认配置使用 Hugging Face 数据集：

- `yahma/alpaca-cleaned`（见 `configs/sft_qwen05b_8g.yaml`）

若要用本地 jsonl：

```text
data/alpaca_sample.jsonl
```

字段至少包含下列之一组合：

- `instruction` + `output`
- 或 `prompt` + `response`
- 或已是 `text` 字段（完整训练文本）

## GRPO / 可验证 reward

使用 `math_sample.jsonl`，每行：

```json
{"prompt": "What is 15 + 27? End with the number.", "answer": "42"}
```

`src/my_lab/rewards.py` 中的 `accuracy_reward` 会对比生成中的最后一个数字与 `answer`。

## DPO

配置默认指向 `trl-lib/ultrafeedback_binarized`（需要 `prompt` / `chosen` / `rejected`）。

后续可换成你自己的偏好数据，仍放本地 jsonl，改 YAML 的 `dataset_path`。

## 下载大文件时

把下载脚本放在 `scripts/`，产物写入 `data/`，并确保：

```gitignore
data/**/*.parquet
# 保留 jsonl 说明与小样例
```
