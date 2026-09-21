# 8GB 显存指南（笔记本 GPU）

> 本文根据 RTX 4060 Laptop **8GB** 场景整理，数字是经验值，不是厂商承诺。  
> 每次跑完请把 **实际峰值** 写回 `experiments/<id>/notes.md`。

## 实测数据（本机 RTX 4060 Laptop 8GB）

以下是本项目**真实跑出来**的数字，不是估算。复现命令见各实验的 `notes.md`。

| 实验 | 底座 | 方法 | seq | batch/accum | peak_allocated | peak_reserved | 耗时 |
|------|------|------|-----|-------------|----------------|---------------|------|
| SFT baseline | Qwen2.5-0.5B | LoRA r=16 + grad ckpt | 512 | 1 / 8 | **1.87 GB** | **4.89 GB** | 345 s (250 步) |
| DPO | Qwen2.5-0.5B | 续 SFT adapter，beta=0.1 | 512 | 1 / 8 | **2.76 GB** | **3.56 GB** | 336 s (125 步) |
| GRPO | Qwen2.5-0.5B | 可验证 reward，num_gen=4 | 128+192 | 1 / 4 | **1.22 GB** | **1.49 GB** | 286 s (400 步) |
| 推理（base 或 +LoRA） | Qwen2.5-0.5B | bf16, max_new_tokens=128 | — | 1 | **0.94–0.95 GB** | **1.00 GB** | — |

### ⚠️ GRPO 不要开 gradient checkpointing

TRL 的 `GRPOTrainer` 会调用 `gradient_checkpointing_enable()` 和
`enable_input_require_grads()`，但**从不在生成前关掉**。实测后果：

| 模型状态 | 生成长度 | 命中 EOS | 输出 |
|----------|---------|---------|------|
| + LoRA / train | 4 | ✅ | `'476'` |
| + LoRA / train / **gradient_checkpointing** | **192** | ❌ | 乱码，永不结束 |

生成被污染 → 奖励恒为 0 → GRPO 完全没有学习信号，而 `loss`/`grad_norm` 都显示 0.0，
看起来像"训练很稳定"。**关掉它即可**：0.5B + LoRA 训练峰值仅 1.22 GB，8G 上根本不需要。
关掉后生成速度还快约 17 倍（原本没有 KV cache，是 O(n²)）。详见
`experiments/2026-09-grpo-math/notes.md`。

### 从实测里学到的三件事

1. **`reserved` 才是"会不会 OOM"的指标，不是 `allocated`。**
   SFT 的 allocated 只有 1.87 GB，但 reserved 到 4.89 GB（占 8 GB 的 61%）。
   只看 allocated 会严重低估风险。`nvidia-smi` 显示的是 reserved 级别。

2. **`reserved` 会随运行时长漂移，不能跨实验直接比。**
   DPO 的 allocated 比 SFT 高 48%（1.87 → 2.76 GB），符合预期（要 forward
   chosen+rejected 两条序列 + 一次参考前向），但它的 reserved 反而更低（3.56 vs 4.89 GB）。
   原因是 caching allocator 的缓存块随步数增长，SFT 跑了 250 步、DPO 只有 125 步。
   → **对比显存时必须固定步数**，否则结论是假的。

3. **LoRA adapter 的推理开销可以忽略。**
   base 0.94 GB → 加 r=16 adapter 0.95 GB，只多约 10 MB。

## 快速对照（能不能塞进去）

| 配置 | 大致结论 |
|------|----------|
| 0.5B bf16 + LoRA，seq=512，bs=1 | 通常可以 |
| 0.5B 全参，seq=512，bs=1 + grad ckpt | 紧但常可行 |
| 1.5B QLoRA / LoRA，seq=512 | 可尝试 |
| 3B LoRA，seq=256–512 | 勉强，易 OOM |
| 7B 任意完整微调 | 本地不要强求 |
| GRPO：num_generations≥4 + 长生成 | 8G 很容易爆 |

## OOM 排查顺序

1. `per_device_train_batch_size=1`
2. 减小 `max_seq_length` / `max_completion_length`
3. 打开 `gradient_checkpointing=true`
4. 开 LoRA；仍不够再考虑 4bit（`bitsandbytes`）
5. 减少 GRPO 的 `num_generations`
6. 降低模型规模（0.5B）
7. 关闭无关进程（浏览器 GPU 加速等）
8. 仍失败：换 WSL2 / 更小数据 / 更小 rank

## 为什么 RL 比 SFT 更吃显存

- SFT：一次 forward + backward（LoRA 时底座可冻结）
- DPO：policy 与 reference 的对比（即便 ref 用 adapter disable）
- GRPO/PPO：每个 prompt 要 **多次采样生成**，生成阶段显存与时间都涨  
  → 8G 上优先短 `max_completion_length`、少 `num_generations`

## 记录模板

在 `experiments/<id>/notes.md` 写：

```markdown
## 环境
GPU: RTX 4060 Laptop 8GB
torch: x.x.x
trl: x.x.x

## 配置摘要
- model: ...
- seq: ...
- batch/accum: ...
- lora: r=...

## 显存
- peak: ? GB（nvidia-smi / torch.cuda.max_memory_allocated）
- 是否 OOM：否 / 第几步

## 结果
- train loss: ...
- 样例生成（前后对比）: ...

## 结论与局限
- ...
```

## 参考计算（粗）

`src/my_lab/memory.py` 里的 `estimate_training_vram_gb` 只用于 **粗估和面试口述**，不能替代实测。
