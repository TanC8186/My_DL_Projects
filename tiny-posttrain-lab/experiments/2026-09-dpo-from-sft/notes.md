# Experiment: 2026-09-dpo-from-sft

## 目标

在 RTX 4060 Laptop 8GB 上，把 SFT 得到的 LoRA adapter 继续用 DPO 做偏好对齐，
验证 `SFT → DPO` 这条后训练链在 8GB 单卡上能否跑通，并记录真实显存与指标。

起点不是原始 instruct 权重，而是 `experiments/2026-09-sft-baseline/output/final`，
这样两次实验构成一条连续的流水线。

## 状态

- [x] 环境安装完成
- [x] 数据确认（UltraFeedback 1000 条切片）
- [x] 训练跑通
- [x] 记录显存峰值
- [x] 验证 adapter 权重确实被更新
- [x] 写出结论与局限

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- torch 2.5.1+cu124 / transformers 4.55.2 / trl 0.21.0 / peft 0.17.1
- 需 `USE_TF=0`、`USE_FLAX=0`、`HF_HUB_OFFLINE=1`（见 `scripts/env.ps1`）

## 配置摘要

- config: `configs/dpo_qwen05b_8g.yaml`
- start_from: `experiments/2026-09-sft-baseline/output/final`（继续训练同一个 adapter）
- 底座: `Qwen2.5-0.5B-Instruct`
- 数据: `trl-lib/ultrafeedback_binarized`，max_samples=1000
- max_prompt_length / max_length: 256 / 512
- batch / accum: 1 / 8 → **125 micro-step = 约 15 次优化器更新**
- lr / beta: 5e-6 / 0.1，loss_type=sigmoid
- 可训练参数: 2,162,688（192 个 LoRA 张量，r=16）

## 显存

- peak_allocated (GB): **2.76**
- peak_reserved (GB): **3.56**
- 是否 OOM: 否

### 与 SFT 对比

| | peak_allocated | peak_reserved |
|---|---|---|
| SFT（0.5B, LoRA, seq512） | 1.87 GB | 4.89 GB |
| DPO（同底座, seq512） | **2.76 GB** | **3.56 GB** |

DPO 的 `allocated` 比 SFT 高约 48%（1.87 → 2.76 GB），符合预期：每步要 forward
`chosen` 和 `rejected` 两条序列，且要额外跑一次参考模型前向（`disable_adapter()`）。

但 `reserved` 反而更低（4.89 → 3.56 GB）。原因是 PyTorch caching allocator 的缓存
随运行时长增长，SFT 跑了 250 步而 DPO 只有 125 步，缓存块更少。
**这说明 reserved 不是稳定的"需求"指标，会随运行长度漂移；对比显存时应固定步数。**

## 结果

| 指标 | 值 |
|------|----|
| train_loss | 0.8226 |
| runtime | 336.09 s |
| steps | 125 micro-step（15 optimizer step）|
| samples/s | 2.975 |
| rewards/accuracies（全程均值） | **0.4550** |
| rewards/accuracies（前半 / 后半） | 0.4687 / **0.4375** |
| rewards/margins（前半 / 后半） | −0.0910 / −0.0771 |
| output | `experiments/2026-09-dpo-from-sft/output/final` |

`rewards/accuracies` 是「chosen 的隐式奖励 > rejected」的比例，随机水平是 0.5。
本次全程 0.455，**低于随机线**，且后半程（0.4375）比前半程（0.4687）更差。

---

## 重大发现：DPO 静默失效（第一次运行训练了 0 个参数）

第一次运行看起来完全正常：125 步跑完、loss 在动、`grad_norm` 稳定在 12–17。
但把产物和输入比对后发现：

```text
SFT adapter SHA256: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
DPO adapter SHA256: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
→ 完全相同，一个字节都没变
```

逐张量比对：192/192 个张量 `max_abs_diff = 0.0`。**这次 DPO 完全没有训练。**

### 根因链（已逐步实测验证）

1. `PeftModel.from_pretrained(model, adapter)` 的 `is_trainable` **默认是 `False`**，
   加载后所有 adapter 参数 `requires_grad=False`：
   ```text
   [is_trainable=False] trainable_params=0        (0.000%)
   [is_trainable=True ] trainable_params=2,162,688 (0.436%)
   ```
2. `Trainer.create_optimizer()` 在训练前收集可训练参数：
   ```text
   [PROBE] create_optimizer called: params_requires_grad=0
   [PROBE] optimizer built with 0 param tensors
   [PROBE] first AdamW.step: n_params=0
   ```
   → **优化器是空的**，`AdamW.step()` 每步都是空转。
3. 训练开始后，参考模型前向会调用 `disable_adapter()`，退出时 `enable_adapter_layers()`
   的**副作用**把 LoRA 的 `requires_grad` 重新打开：
   ```text
   [PROBE] total=482 requires_grad=192 with_grad=192 norm=33.9176
   ```
4. 于是梯度真的产生了（`grad_norm` 非零、loss 会动），**但优化器里没有参数**，
   权重永远不变。

### 为什么这个 bug 危险

- **loss 会下降、`grad_norm` 非零**，日志上看不出任何异常。
- `Trainer` 不会报错：`AdamW([])` 合法，空 step 合法。
- 只有把**产物和输入做字节比对**才会暴露。

`grad_norm` 在这里是**误导性指标**：它只证明"有梯度"，不证明"有参数被更新"。

### 修复

```python
model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
```

并在脚本里加防呆检查，让同类错误无法静默发生：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable == 0:
    raise RuntimeError("No trainable parameters. ...")
```

修复后验证：

```text
[info] trainable parameters: 2,162,688
tensors_changed = 192/192
max_abs_diff    = 2.040372e-04
```

---

## 另外两个坑

### 1. 参考模型不是 SFT 模型，而是原始 instruct 模型

`ref_model=None` 时，TRL 走 `null_ref_context()`：

```python
with (self.accelerator.unwrap_model(self.model).disable_adapter()
      if self.is_peft_model and not self.ref_adapter_name
      else nullcontext()):
```

`disable_adapter()` 禁用**所有** LoRA，所以参考 logits 来自 **base Qwen2.5-0.5B-Instruct**，
而不是 SFT 之后的模型。也就是说，本次 DPO 的 KL 约束把模型往"原始 instruct"拉，
**部分抵消了 SFT 的适配**。

标准 `SFT → DPO` 流水线要求参考模型 = SFT 模型。要满足有两条路：

1. 先把 SFT adapter merge 进底座，再用 merge 后的模型作为参考，另起一个新 LoRA 做 DPO；
2. 用 TRL 的多 adapter 机制：`ref_adapter_name="sft"` + `model_adapter_name="dpo"`
   （`DPOConfig` 里确实有这两个字段）。

本次两条都没用，所以这是一个**已知的方法论缺陷**，结论解释时需要考虑进去。

### 2. `logps` 是求和，长度有偏

```python
all_logps = per_token_logps[:, 1:].sum(-1)   # 求和，不是平均
```

所以 `logps/chosen = -425.5` vs `logps/rejected = -321.0` 主要反映**长度差**（rejected 更短），
不能直接当作"偏好"。实测同一批数据里 `logps/chosen` 在 −237 到 −446 之间跳。

另外，TRL 0.21.0 的 `DPOConfig` **没有** `average_log_prob` 这个长度归一化开关
（它只作为 liger kernel 的内部参数硬编码为 `False`），所以在这个版本里想归一化长度并不直接。

## 结论与局限

- `SFT → DPO` 链路在 8GB 上**跑得通**，显存 2.76 GB allocated / 3.56 GB reserved。
- 本次 DPO **没有产生可测量的偏好提升**：`rewards/accuracies` 0.455，低于随机线。
- 主要原因是**训练量太小**：1000 条样本 / bs1 / accum8 只有约 **15 次优化器更新**，
  lr=5e-6 下 adapter 的最大变化仅 `2.04e-04`。这么小的步数不足以改变偏好。
- 其次是上面两个方法论缺陷（参考模型错位、logps 长度偏差）会稀释信号。
- 局限：仅 0.5B、仅 1000 条 UltraFeedback、单次运行、未做 held-out 偏好评测。
  不能据此断言"DPO 无效"，只能说**本次配置规模不足以产生效果**。

## 下一步（若要继续）

1. 提高优化器步数：`max_samples` 提到 5000+，或 `gradient_accumulation_steps` 降到 2。
2. 修正参考模型：merge SFT adapter 后再做 DPO，或改用 `ref_adapter_name`。
3. 用 `generate.py --compare` 对比 SFT 与 DPO 的生成结果，而不是只看训练指标。
