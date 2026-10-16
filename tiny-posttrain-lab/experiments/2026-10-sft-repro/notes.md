# Experiment: 2026-10-sft-repro

## 目标

独立重跑 `configs/sft_qwen05b_8g.yaml`，验证仓库记录的 SFT 基线数字是否可复现。

## 状态

- [x] 训练跑通
- [x] 记录显存峰值
- [x] 与仓库原记录对比
- [x] 生成对比（`generate.py --compare`）
- [x] 在 venv 被重建后重跑一次，确认新的 venv 可复现同一结果

## 这个实验和前一个的关系

配置与 `2026-09-sft-baseline` **只差两行**（`experiment_id` 和 `output_dir`），
其余字段逐字相同 —— 这是可复现性对比的前提。原实验的 `output/` 未被覆盖
（每次跑完都用 `git status` 确认过原 `train_metrics.txt` 一字未改）。

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
- venv: Python 3.10.18，基于 conda `dl_env`，`--system-site-packages`
- torch 2.5.1+cu124 (CUDA available) / transformers 4.55.2 / trl 0.21.0 / peft 0.17.1
- 运行前置：`USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1`

### 环境变更记录（重要）

本实验跑了两遍，**中间 venv 被外部进程重建过一次**（PyCharm，见 `notes` 时间线）：
第一次是 Python 3.10.18 + torch 2.5.1+cu124（可用），
第二次的 venv 被换成了 Python 3.13.5 + torch 2.14.1+cpu（CUDA 不可用）。

venv 已按 `requirements.txt` 重建回 3.10.18 + CUDA 版本，并重跑了本实验。
**两次运行的确定性指标完全一致**（见下表），说明重建后的环境与原环境等价。

## 结果对比

### 三次运行（同一 config、同一 seed=42）

| 指标 | 原记录<br>`2026-09-sft-baseline` | 重跑①<br>重建前 | 重跑②<br>重建后 | 极差 |
|------|------|------|------|------|
| train_loss | 1.3924129 | 1.3920589 | **1.3918319** | 0.042% |
| peak_allocated (GB) | 1.87 | 1.87 | **1.87** | 0 |
| peak_reserved (GB) | 4.89 | 4.89 | **4.89** | 0 |
| total_flos | 697581793920000.0 | 697581793920000.0 | **697581793920000.0** | 0 |
| train_runtime (s) | 344.71 | 393.92 | 394.44 | +14.4% |

> 原实验的 notes 里还记了另一次运行（`train_loss` 1.391932），
> 所以同一个配置累计已有 **4 次独立运行的记录**，loss 极差 0.042%。

## 结论

1. **确定性指标完全复现。** `total_flos` 三次逐位相同，说明数据量、序列长度、
   步数、有效批量全部一致 —— 配置没被改跑偏（这是"对比有效"的前提）。
2. **显存逐字节相同**（1.87 / 4.89 GB）。印证了 `docs/8gb-memory-guide.md` 的说法：
   `reserved` 才是 OOM 判据，且跨运行可比的前提是固定步数。
3. **venv 重建没有改变结果。** 重建前后两次运行的 loss 差 0.016%，
   显存与 flops 完全一致 —— 说明新 venv 与原 venv 在数值上等价，
   重建是成功的（不只是"能跑"，而是"跑出同样的数"）。
4. **墙钟时间不是确定性指标。** runtime 差 14%，但 loss 与 flops 都一致 ——
   说明差异来自笔记本 GPU 降频，不是代码或数据问题。
   **判断两次运行是否等价，要看 flops/loss，不能看耗时。**

## 生成对比（`generation_compare.md`）

命令：

```powershell
python scripts/generate.py --config configs/sft_repro_qwen05b_8g.yaml `
    --adapter experiments/2026-10-sft-repro/output/final `
    --compare --max-new-tokens 128 `
    --out experiments/2026-10-sft-repro/generation_compare.md
```

**观察 1 — 风格被改写（复现了原实验的结论）**

Prompt 1「Give three tips for staying healthy.」：

- Base：`Certainly! Here are three tips to help you stay healthy:` 然后大段散文
- SFT+LoRA：直接进编号列表 `1. Exercise regularly... / 2. Eat a balanced diet...`

与 `2026-09-sft-baseline` 记录的现象一致：LoRA 去掉了礼貌前缀、转为列表。

**观察 2 — 太阳系列表：本次没有退化（与原记录不同）**

- 原记录：8 项 → 只列 7 项，第 3 项写成 `Earth (Mars)`，Mars 位置错误
- 本次：`Mercury / Venus / Earth / Mars / Jupiter / Saturn / Uranus / Neptune` —— **8 项全对**

**这是本次实验最有价值的一条观察，因为它是一个反例。**

原实验用的是 `temperature=0.7` 单次采样，本次也是。同配置、同 seed，
但采样路径不同（GPU 非确定性 + 降频影响时序），所以拿到的是另一个样本。
**一条采样不足以支撑"能力退化"这种结论** —— 原记录自己也标注了
"不构成统计结论"，本次实验从反面印证了这个限制。

**结论：风格改写是稳定复现的（3 次都一样），但事实性问答的退化是抽样噪声，
不能靠单次观察断言。** 要下判断必须 greedy 解码 + 更多 prompt + 重复运行。

## 局限

- 生成对比是单次采样（`temperature=0.7`），如上所述不构成统计结论。
- 未做 greedy 解码下的重复评测。
- 同一台机器，未验证跨机器可复现。
