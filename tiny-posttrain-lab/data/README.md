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

数据集由脚本生成，答案用 Python 计算而非手写：

```powershell
python scripts/make_math_dataset.py --n 300 --eval-n 100 --seed 42 `
    --out data/math_train.jsonl --eval-out data/math_eval.jsonl
```

- `math_train.jsonl`（200 行）— 训练
- `math_eval.jsonl`（100 行）— **held-out 评测**，与训练集**重叠数为 0**
- `math_sample.jsonl`（10 行）— 早期手写小样例，保留作最小可读示例

两个 split 从同一池子确定性切分，所以不会重叠。用两个独立 seed 各生成一份会出现重复题，
那等于在训练集上评测 —— 这一点在 `experiments/2026-09-grpo-math/notes.md` 里有记录。

每行格式：

```json
{"prompt": "What is 15 + 27? End with the number only.", "answer": "42"}
```

`src/my_lab/rewards.py` 中的 `accuracy_reward` 会取生成文本里的**最后一个数字**与 `answer` 比对。

> ⚠️ 这个 reward 很脆弱：模型若没遵守「只以数字结尾」，取到的可能是推理过程里的中间数字。
> 例：`9 * 9` 的输出被解析成 `1.0`。这是 reward misspecification 的典型例子，
> 换任务时务必先确认 checker 本身正确。

评测用 `scripts/eval_math.py`（greedy 解码，保证可复现）：

```powershell
python scripts/eval_math.py --config configs\grpo_qwen05b_8g.yaml `
    --data data\math_eval.jsonl --greedy --tag base
```

## DPO

配置默认指向 `trl-lib/ultrafeedback_binarized`（需要 `prompt` / `chosen` / `rejected`）。

后续可换成你自己的偏好数据，仍放本地 jsonl，改 YAML 的 `dataset_path`。

## 下载大文件时

把下载脚本放在 `scripts/`，产物写入 `data/`，并确保：

```gitignore
data/**/*.parquet
# 保留 jsonl 说明与小样例
```
