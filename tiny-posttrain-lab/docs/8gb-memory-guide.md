# 8GB 显存指南（笔记本 GPU）

> 本文根据 RTX 4060 Laptop **8GB** 场景整理，数字是经验值，不是厂商承诺。  
> 每次跑完请把 **实际峰值** 写回 `experiments/<id>/notes.md`。

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
