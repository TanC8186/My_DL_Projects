# 05 · DPO 训练脚本 `scripts/train_dpo.py`

这一篇讲 DPO。相比 SFT，它多了两个新概念（**参考模型** 和 **偏好对**），
也藏着全项目**最隐蔽的一个 bug**。

---

## 0. DPO 是什么（先建立直觉）

### SFT 和 DPO 的根本区别

| | SFT | DPO |
|---|---|---|
| 数据 | `(prompt, 好的回答)` | `(prompt, 好回答, 坏回答)` |
| 目标 | 最大化好回答的概率 | **让好回答的概率相对坏回答上升** |
| 监督信号 | "这个答案是对的，学它" | "这两个答案里，A 比 B 好" |

**关键洞察：DPO 不需要"标准答案"，只需要"哪个更好"。**
这让数据获取容易得多 —— 让模型生成两个回答，人类标注哪个好，比让人从零写一个标准答案便宜。

### DPO 的核心公式

```text
L_DPO = -log σ( β · [ log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ] )

其中：
  y_w = chosen（更好的回答）
  y_l = rejected（更差的回答）
  π_θ = 当前策略（正在训练的模型）
  π_ref = 参考模型（冻结的起点）
  β   = 温度系数，控制对参考模型的偏离惩罚
  σ   = sigmoid 函数
```

**直觉解释**：括号里是"好答案相对坏答案的对数概率比，减去参考模型的同一个比值"。

- 减去 `π_ref` 是为了**消除"哪个答案本身更容易生成"这个混杂因素**
- 括号里的值越大 → 模型越倾向于好答案 → `σ(·)` 越接近 1 → `-log` 越接近 0（loss 小）

**TRL 的实现**（`dpo_trainer.py:1042-1046`）：

```python
logratios = chosen_logps - rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = logratios - ref_logratios
```

**注意 `beta` 不在 `logits` 里**，而是在算 reward 时乘上（`dpo_trainer.py:1111-1112`）：

```python
chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
```

所以日志里的 `rewards/chosen` 就是论文中的隐式奖励 `β·log(π_θ/π_ref)`。

> 完整推导和源码行号见 [`docs/trl-reading-notes.md`](../trl-reading-notes.md)。

---

## 1. 脚本结构：与 SFT 的差异

```text
train_dpo.py
├── 引导段（与 SFT 完全相同）
├── main()
│   ├── 【新增】--adapter 参数 + start_from 解析
│   ├── 模型段（+ 【新增】加载已有 adapter + 可训练参数守卫）
│   ├── 数据段（用 HF 数据集，不做 format_example）
│   ├── 【新增】peft_config 的二选一逻辑
│   ├── 训练段（DPOConfig，多了 beta / loss_type / max_prompt_length）
│   └── 落盘段（与 SFT 相同）
```

**没有 `build_lora_config` 和 `load_dataset_from_cfg` 这两个辅助函数** ——
DPO 的数据加载逻辑简单到不需要抽函数（直接调 `load_dataset`），
LoRA 配置也内联在 `main()` 里（因为它要和"是否传了 adapter"耦合判断）。

> **这是刻意的**：抽函数的阈值是"被复用"或"逻辑复杂"。DPO 的 LoRA 配置
> 只出现一次且依赖上下文，抽出去反而增加跳转成本。

---

## 2. 新增的第一个概念：`start_from` / `--adapter`

### 参数定义

```python
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Existing LoRA adapter to continue training (e.g. the SFT output). "
            "Defaults to the config's `start_from`; omit both to start a fresh "
            "adapter from the config's LoRA settings."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # `start_from` in the YAML lets the SFT -> DPO chain be reproducible without CLI flags.
    adapter = args.adapter or cfg.get("start_from")
```

### `args.adapter or cfg.get("start_from")` —— 优先级设计

```text
命令行 --adapter  >  配置 start_from  >  None（起新 adapter）
```

**为什么这么设计？**

| 场景 | 用什么 | 理由 |
|------|--------|------|
| 正式实验 | 配置里的 `start_from` | **配置是记录**。别人 clone 你的仓库跑同一条命令，得到同样的结果 |
| 临时试验 | 命令行 `--adapter` | 不想为一次尝试改配置文件 |

**如果只有命令行参数**：每次跑都要打一长串路径，而且 `notes.md` 里记录的
"这次实验怎么跑的"就丢了。
**如果只有配置**：想试个别的 adapter 就得改配置，容易忘记改回来。

**两者都要，且命令行优先** —— 这是 CLI 工具的标准设计。

### 加载 adapter（⚠️ 这里有坑）

```python
    if adapter:
        from peft import PeftModel

        # is_trainable=True is REQUIRED. The default (False) leaves every adapter
        # parameter with requires_grad=False, so Trainer.create_optimizer() collects
        # an empty parameter list. Training then looks healthy (loss moves, grad_norm
        # is non-zero, because the reference forward's disable_adapter()/
        # enable_adapter_layers() side effect flips requires_grad back on) while
        # AdamW.step() updates nothing and the saved adapter is byte-identical.
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        print(f"[info] continuing from adapter: {adapter}")
```

**这 9 行注释是整个项目最值钱的一段文字。** 它记录了一个让训练完全失效、
但所有指标都显示"正常"的 bug。

---

## 3. ⚠️ 核心案例：`is_trainable` 静默失效 bug

### 3.1 现象

第一次跑 DPO，一切看起来正常：

```text
[info] continuing from adapter: .../2026-09-sft-baseline/output/final
[info] dataset rows=1000 columns=['chosen', 'rejected', 'score_chosen', 'score_rejected']
[info] adapter given -> peft_config=None (keep training the loaded adapter)
  0%|          | 0/125 [00:00<?, ?it/s]
  1%|          | 1/125 [00:04<08:17,  4.01s/it]
  ...
{'loss': 0.826, 'grad_norm': 12.684, 'rewards/accuracies': 0.575, ...}
{'loss': 0.6933, 'grad_norm': 16.956, 'rewards/accuracies': 0.55, ...}
...
[info] peak VRAM: peak_allocated=2.75 GB, peak_reserved=3.54 GB
[done] model   -> .../output/final
```

**125 步跑完、loss 在动、grad_norm 非零、显存正常、没有报错。**

### 3.2 发现过程：把产物和输入做字节比对

```powershell
$sft = (Get-FileHash .../2026-09-sft-baseline/output/final/adapter_model.safetensors).Hash
$dpo = (Get-FileHash .../2026-09-dpo-from-sft/output/final/adapter_model.safetensors).Hash
```

```text
SFT: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
DPO: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
RESULT: IDENTICAL - weights did NOT change!
```

**逐张量比对**：

```text
[DPO-final] max_abs_diff=0.000000e+00  tensors_changed=0/192
```

**192 个张量，一个字节都没变。这次 DPO 完全没有训练。**

> **这是全项目最重要的一条方法论：指标只是线索，产物才是证据。**
> `loss` 在降、`grad_norm` 非零 —— 两个最常被用来判断"训练是否正常"的指标
> **都骗了你**。

### 3.3 根因链（每一步都实测验证过）

#### 第 1 步：`is_trainable` 默认是 `False`

```python
# 实测对比
[is_trainable=False] total=496,195,456  trainable_params=0        (0.000%)
[is_trainable=True ] total=496,195,456  trainable_params=2,162,688 (0.436%)
```

**PEFT 的 `PeftModel.from_pretrained` 签名**：

```python
@classmethod
def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, ...)
```

**`is_trainable=False` 是默认值。** 设计意图是"加载 adapter 通常是为了推理"，
所以默认按推理模式加载（不建梯度图、参数 `requires_grad=False`）。

**对推理是对的，对续训是灾难。**

#### 第 2 步：`Trainer` 建了一个**空**优化器

```python
# 拦截 Trainer.create_optimizer 的实测输出
[PROBE] create_optimizer called: params_requires_grad=0
[PROBE] optimizer built with 0 param tensors
[PROBE] first AdamW.step: n_params=0
```

**`Trainer.create_optimizer()` 的源码逻辑**（transformers）：

```python
params = [p for p in self.model.parameters() if p.requires_grad]
optimizer = optimizer_cls(params, **optimizer_kwargs)
```

**`requires_grad` 全为 `False` → `params` 是空列表 → `AdamW([])` 是合法的空优化器。**

**这是这个 bug 最阴险的地方：`torch.optim.AdamW([])` 不报错。**
它创建一个 `param_groups = [{'params': []}]` 的优化器，`step()` 什么都不做。

#### 第 3 步：参考模型的前向**副作用**把梯度又打开了

这是最反直觉的一步。

```python
# 在真实 DPO 一步训练中拦截 clip_grad_norm_
[PROBE] total=482 requires_grad=192 with_grad=192 norm=33.9176
```

**192 个参数变成了 `requires_grad=True`，而且真的有梯度（norm=33.9）！**

**谁改的？** 不是 TRL 显式改的。TRL 的 DPO 用 `null_ref_context()` 拿参考 logits：

```python
# dpo_trainer.py:890-894
def null_ref_context(self):
    with (self.accelerator.unwrap_model(self.model).disable_adapter()
          if self.is_peft_model and not self.ref_adapter_name
          else nullcontext()):
        ...
```

`disable_adapter()` 是 PEFT 的上下文管理器，它**临时关闭 LoRA**：

```python
# PEFT 内部（简化）
@contextmanager
def disable_adapter(self):
    try:
        self.base_model.disable_adapter_layers()   # 关闭：把 adapter 参数设为 requires_grad=False
        yield
    finally:
        self.base_model.enable_adapter_layers()    # 恢复：把 adapter 参数设回 requires_grad=True ← 副作用！
```

**`enable_adapter_layers()` 无条件把 `requires_grad` 设回 `True`** ——
它不知道这个 adapter 本来是冻结的。

**时序是致命的：**

```text
t0: PeftModel.from_pretrained(is_trainable=False)
      → 所有 adapter 参数 requires_grad=False

t1: Trainer.train() → create_optimizer()
      → 收集 [p for p in params if p.requires_grad] = []   ← 空！
      → 优化器建立，含 0 个参数

t2: 第一步 training_step
      → 需要参考 logits → disable_adapter() ... enable_adapter_layers()
      → 副作用：adapter 参数 requires_grad 变回 True

t3: backward()
      → 梯度真的产生了（grad_norm 非零）

t4: optimizer.step()
      → 优化器里 0 个参数 → 什么都不更新

循环 t2~t4 共 125 次 → 权重始终不变
```

**优化器在 t1 就已经定型了，而参数在 t2 才被"解锁"。**

### 3.4 为什么这个 bug 特别危险

| 表现 | 为什么骗过了检查 |
|------|----------------|
| `loss` 正常下降 | loss 是前向算出来的，前向是正常的（adapter 权重没变，但 loss 本身有值） |
| `grad_norm` 非零 | **梯度确实产生了** —— 只是优化器没接收 |
| 显存正常 | 参考前向确实在跑，该占的显存都占了 |
| 没有异常/警告 | `AdamW([])` 合法、空 `step()` 合法 |
| 训练速度正常 | 每步的 forward/backward 都真的执行了 |

**唯一能暴露它的方法**：把产物和输入做**字节级比对**。

### 3.5 修复

```python
model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
```

**修复后验证**：

```text
[info] trainable parameters: 2,162,688
tensors_changed = 192/192
max_abs_diff    = 2.040372e-04
mean_abs_diff   = 1.585718e-04
```

**192/192 张量全部改变，最大差异 2.04e-4。**

> **为什么差异这么小？** 因为 `lr=5e-6` + 只有约 15 次优化器更新
> （125 micro-step / accum 8）。**这个"改动幅度"后来成了分析"为什么没效果"的关键证据** ——
> 见 `experiments/2026-09-dpo-from-sft/notes.md`。

### 3.6 防呆守卫：让同类错误无法再静默发生

```python
        # Only meaningful on this path. When TRL attaches a fresh LoRA itself, the
        # count here would just be the (still unfrozen) base model.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[info] trainable parameters: {trainable:,}")
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters. The optimizer would be empty and training "
                "would silently do nothing. Check the adapter / LoRA configuration."
            )
```

**设计要点：**

#### 1. 只检查 `trainable == 0`，不检查"数量是否合理"

因为"合理的数量"取决于 `r` 和 `target_modules`，写死一个阈值会误报。
**零是一个绝对、无歧义的信号** —— 零个可训练参数必然是错误。

#### 2. 用 `RuntimeError` 而不是 `assert`

`assert` 在 Python 优化模式（`-O`）下**会被跳过**。**安全检查必须用 `raise`。**

#### 3. 注释解释了"为什么放在这个位置"

```python
# Only meaningful on this path. When TRL attaches a fresh LoRA itself, the
# count here would just be the (still unfrozen) base model.
```

**这个注释是必要的**，因为守卫放在 `if adapter:` **内部**看起来像是遗漏 ——
为什么只在传了 adapter 时才检查？

原因：**不传 adapter 时，`peft_config` 交给 TRL，TRL 会自己包装模型并冻结底座。
在那一行代码的位置，模型还是"原始底座"（全部参数可训练，494M），
所以计数毫无意义。**

实测证据（GRPO 脚本的 smoke test 输出）：

```text
[info] trainable parameters: 494,032,768     ← 这是整个底座，不是 LoRA！
```

**所以守卫必须放在"adapter 已经手动挂好"的那条路径上。**

> **这个细节是迭代出来的**：第一版守卫放在 `if adapter:` 外面，
> 打印出 494M 这个误导性数字，才意识到位置错了。

---

## 4. 新增的第二个概念：`peft_config` 的二选一

```python
    peft_config = None
    lora = cfg.get("lora") or {}
    if adapter:
        # The adapter is already attached to the model. TRL obtains the reference
        # logits by calling disable_adapter(), so no second model copy is needed.
        print("[info] adapter given -> peft_config=None (keep training the loaded adapter)")
    elif lora.get("enabled"):
        peft_config = LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )
```

### 这个 `if/elif` 的逻辑

```text
传了 adapter？
├── 是 → peft_config = None     ← 因为 adapter 已经手动挂上了
└── 否 → 看 YAML 的 lora.enabled
         ├── true  → 建新的 LoraConfig（TRL 会包装模型）
         └── false → peft_config = None（全参训练）
```

**为什么传了 adapter 就要 `peft_config = None`？**

因为**两种方式不能叠加**：

- 传 `peft_config`：TRL 内部调 `get_peft_model(base_model, peft_config)` → **创建一个新 adapter**
- 手动 `PeftModel.from_pretrained`：adapter **已经挂上了**

如果两个都做，结果是在已挂 adapter 的模型上再套一层 → 行为未定义。

**注意 `elif` 而不是第二个 `if`** —— 保证两条路径互斥。

### 注释里的关键信息：为什么不需要第二个模型

```python
# TRL obtains the reference logits by calling disable_adapter(), so no second
# model copy is needed.
```

**这是 DPO 显存优化的核心。** 传统 DPO 实现要加载两份模型（策略 + 参考），
显存翻倍。TRL 用 PEFT 的技巧避免了：

```text
参考 logits 需要 π_ref = 关闭 adapter 的模型
        ↓
PeftModel 支持 disable_adapter() 临时关闭 LoRA
        ↓
同一个模型实例，跑两次前向：
  第一次：adapter 开启 → π_θ 的 logits
  第二次：adapter 关闭 → π_ref 的 logits
        ↓
省下了一整份底座权重（0.5B = 1 GB）
```

**代价**：`disable_adapter()` 会触发 `enable_adapter_layers()` 的副作用 ——
**这就是上面那个 bug 的成因。一个优化手段引入了另一个 bug。**

实测显存印证：DPO `peak_allocated = 2.76 GB`。
如果加载两份模型，至少是 `1.0（底座）+ 1.0（参考）= 2.0 GB` 只是权重，
加上激活和优化器会远超 2.76 GB。

### 全参训练的分支

```python
elif lora.get("enabled"):
```

如果 `lora.enabled: false` 且没传 adapter，`peft_config` 保持 `None` →
**TRL 会做全参微调**。

**0.5B 全参在 8GB 上能跑吗？** 参考 `docs/8gb-memory-guide.md`：
`0.5B 全参，seq=512，bs=1 + grad ckpt | 紧但常可行`。
但优化器状态就要约 4 GB（`0.5B × 4 字节 × 2`），加上权重和激活会很紧张。

---

## 5. 数据段：与 SFT 的差异

```python
    ds = load_dataset(data_cfg["dataset_name"], split=data_cfg.get("dataset_split", "train"))
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    print(f"[info] dataset rows={len(ds)} columns={ds.column_names}")
```

**比 SFT 简单**：直接 `load_dataset`，没有 `format_example`。

**为什么？** 因为 DPO 的数据格式是**结构化的**（`chosen` / `rejected` 两列），
不需要拍平成文本。TRL 的 `DPOTrainer` 会自己处理 tokenization。

实测列名：

```text
[info] dataset rows=1000 columns=['chosen', 'rejected', 'score_chosen', 'score_rejected']
```

### ⚠️ 注意：没有 `prompt` 列

`ultrafeedback_binarized` 的结构是**对话式**的：

```json
{
  "chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "好的回答"}],
  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "差的回答"}],
  "score_chosen": 8.0,
  "score_rejected": 3.0
}
```

**prompt 藏在 `chosen` / `rejected` 的消息列表里**（两者共享的第一条 user 消息）。

TRL 用 `maybe_extract_prompt` 自动抽出来（`dpo_trainer.py:630`）：

```python
dataset = dataset.map(maybe_extract_prompt, **map_kwargs)
```

**这个"自动抽取"是有前提的**：它假设 `chosen` 和 `rejected` 有**共同的前缀**
（即同一个 prompt）。如果数据里两者的 prompt 不同，抽出来的 prompt 是错的 ——
**而且不会报错。**

> **可讲的点**：这是一个"隐式约定"。TRL 的文档说明了这个行为，
> 但代码不校验。**用公开数据集时要知道它在做什么，不能盲信。**

---

## 6. 训练段：`DPOConfig` 的差异参数

```python
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        beta=float(train_cfg.get("beta", 0.1)),
        loss_type=str(train_cfg.get("loss_type", "sigmoid")),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 256)),
        max_length=int(data_cfg.get("max_length", 512)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        report_to=list(train_cfg.get("report_to") or []),
        seed=int(train_cfg.get("seed", 42)),
    )
```

### 6.1 `beta` —— DPO 最重要的超参

```python
beta=float(train_cfg.get("beta", 0.1)),
```

**`beta` 控制"允许偏离参考模型多少"：**

| `beta` | 效果 |
|--------|------|
| 大（如 0.5） | 严格贴近参考模型。安全但学得慢 |
| **0.1（论文默认）** | 平衡 |
| 小（如 0.01） | 允许大幅偏离。学得快但可能训崩、遗忘 |
| 0 | 完全没有约束 —— **等价于不减去参考模型** |

**数学上看 `beta` 在哪**：

```text
L = -log σ( β · (logratio_policy − logratio_ref) )

β 越小 → 括号里的值越被压缩 → σ 越接近 0.5 → 梯度信号越弱但越平滑
β 越大 → 对"偏离参考"的惩罚越强
```

**`beta` 同时出现在 loss 的 sigmoid 里面，和 reward 的计算里**：

```python
# dpo_trainer.py:1111-1112
chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
```

所以日志里的 `rewards/chosen` 已经被 `beta` 缩放过。
**比较不同 `beta` 实验的 reward 时要注意这一点** —— 数值不可直接比。

### 6.2 `loss_type` —— DPO 有很多变体

```python
loss_type=str(train_cfg.get("loss_type", "sigmoid")),
```

TRL 0.21.0 支持的选项（来自 `DPOConfig` 的 docstring）：

| 值 | 来源 | 特点 |
|----|------|------|
| `sigmoid` | 原始 DPO 论文 | 默认，标准做法 |
| `hinge` | 合页损失 | 对极端样本更鲁棒 |
| `ipo` | IPO 论文 | 用平方损失替代 log-sigmoid，更稳定 |
| `bco_pair` | BCO 论文 | 用成对二元分类 |
| `kto_pair` | KTO 论文 | 只需要"好/坏"标签，不需要成对 |
| `apo_zero` / `apo_down` | APO 论文 | 非对称处理 |
| `discopop` | DiscoPOP 论文 | 对数比率调制的 loss |
| `sft` | — | **退化成普通 SFT loss**（只学 chosen） |
| `robust` | — | 对噪声标注更鲁棒 |

**多个可以组合**（`loss_type=["sigmoid", "sft"]`），
配合 `loss_weights` 加权。TRL 内部用 `logits` 和 `losses` 累加实现。

**`loss_type="sft"` 是个有趣的选项** —— 它让 DPO 退化成"只学 chosen"，
可以用来做消融实验：**"到底是偏好学习起了作用，还是只是多学了一遍好答案？"**

> **本项目没做这个消融**，这是一个可以继续做的方向。

### 6.3 `max_prompt_length` / `max_length`

```python
max_prompt_length=int(data_cfg.get("max_prompt_length", 256)),
max_length=int(data_cfg.get("max_length", 512)),
```

**DPO 有两个长度预算**（SFT 只有一个）：

```text
max_prompt_length = 256   ← 问题部分最多 256 token
max_length        = 512   ← prompt + completion 总共最多 512 token
                          → completion 的预算是 512 - 256 = 256
```

**为什么 prompt 和 completion 要分开限？**

因为 DPO 的 loss 只关心 **completion 部分**的对数概率
（`chosen_logps` 是 completion 的 logps，不含 prompt）。
如果 prompt 太长挤掉了 completion 的预算，就没有有效的监督信号了。

**TRL 内部怎么处理超长？** 有 `truncation_mode` 参数（默认 `"keep_start"`），
决定超长时保留开头还是结尾。对于 completion 更重要的场景，
`"keep_end"` 可能更合适。

---

## 7. 构造 Trainer：`ref_model=None` 的深意

```python
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter: TRL can use disabled adapter as ref
        args=dpo_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
```

### `ref_model=None` 触发了什么

看 TRL 源码（`dpo_trainer.py:908-911`）：

```python
with torch.no_grad(), compte_ref_context_manager:
    if self.ref_model is None:
        with self.null_ref_context():
            ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
    else:
        ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
```

`ref_model=None` → 走 `null_ref_context()` → `disable_adapter()` → **参考模型 = 关闭 LoRA 的底座**。

### ⚠️ 这是一个**方法论缺陷**，不是 bug

**标准 `SFT → DPO` 流水线要求：**

```text
参考模型 = SFT 之后的模型（冻结）
```

**但本项目实际得到的是：**

```text
参考模型 = 原始 Qwen2.5-0.5B-Instruct（所有 LoRA 被禁用）
```

**后果**：DPO 的 KL 约束把模型往**原始 instruct 权重**拉，
**部分抵消了 SFT 的适配**。

```text
        原始 instruct
             │
             │  SFT（alpaca）
             ▼
        SFT adapter ──────┐
             │            │  我们希望参考模型在这里
             │  DPO       │
             ▼            ▼
        期望：DPO 相对 SFT 做微调
        实际：DPO 相对原始 instruct 做调整  ← 方向被拉回去了
```

### 两种正确的做法

**做法 1：先 merge 再 DPO**

```python
# 把 SFT adapter 合并进底座，得到一个"真的 SFT 模型"
merged = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()
merged.save_pretrained("sft_merged")
# 然后用 merged 作为 base，另起一个新 LoRA 做 DPO
```

此时 `disable_adapter()` 关闭的是**新的 DPO adapter**，
参考模型就是 `merged`（= SFT 模型）。✅

**做法 2：用多 adapter（TRL 原生支持）**

`DPOConfig` 里有这两个字段（`dpo_config.py:229, 233`）：

```python
model_adapter_name: Optional[str] = field(default=None, ...)
ref_adapter_name: Optional[str] = field(default=None, ...)
```

用法：把 SFT adapter 挂成名为 `"sft"` 的参考 adapter，
再挂一个可训练的 `"dpo"` adapter，然后设 `ref_adapter_name="sft"`。

TRL 的 `null_ref_context()` 支持这个：

```python
def null_ref_context(self):
    with (self.accelerator.unwrap_model(self.model).disable_adapter()
          if self.is_peft_model and not self.ref_adapter_name      # ← 没有指定 ref adapter 才禁用全部
          else nullcontext()):
        if self.ref_adapter_name:
            self.model.set_adapter(self.ref_adapter_name)          # ← 切换到指定的参考 adapter
        yield
        if self.ref_adapter_name:
            self.model.set_adapter(self.model_adapter_name or "default")
```

**本项目两条都没用** → 所以这是一个**已知的方法论缺陷**，
在 `notes.md` 里明确记录了。

> **这就是为什么实验记录里必须写"局限"**：如果只写"DPO 跑通了，accuracy 0.455"，
> 读者会以为这是标准 DPO 的结果。写了参考模型错位，读者才知道这个数字
> 不能和论文对比。

---

## 8. 实测结果与解读

### 8.1 最终指标

```text
{'train_runtime': 336.0944, 'train_samples_per_second': 2.975,
 'train_steps_per_second': 0.372, 'total_flos': 0.0,
 'train_loss': 0.8225601577758789, 'epoch': 1.0}
peak_allocated=2.76 GB, peak_reserved=3.56 GB
```

### 8.2 关键指标：`rewards/accuracies`

```text
logged steps: 25
accuracy  overall mean : 0.4550      ← 低于随机线 0.5！
accuracy  first half   : 0.4687
accuracy  last half    : 0.4375      ← 越训越差
margins   first half   : -0.0910
margins   last half    : -0.0771
```

**`rewards/accuracies` 是什么？**

TRL 的定义（`dpo_trainer.py:1694`）：

```python
reward_accuracies = (chosen_rewards > rejected_rewards).float()
```

即"**模型给 chosen 的隐式奖励高于 rejected 的比例**"。

**随机水平是 0.5**（瞎猜）。**0.455 低于随机**。

### 8.3 为什么没有提升（三个原因，按重要性排序）

#### 原因 1：训练量太小（主因）

```text
1000 条样本 / batch 1 / accum 8 = 125 次 micro-step
125 / 8 = 约 15 次优化器更新
```

**15 次更新对 DPO 来说几乎等于没训。** 证据是改动幅度：

```text
tensors_changed = 192/192
max_abs_diff    = 2.040372e-04      ← 权重只移动了 2e-4
```

`lr=5e-6` × 15 步 → 累积更新量就是这个量级。**这么小的改动不可能改变偏好行为。**

#### 原因 2：参考模型错位（方法论缺陷）

见第 7 节。KL 项把模型往原始 instruct 拉，与 SFT 的适配方向冲突。

#### 原因 3：`logps` 的长度偏差

```python
# dpo_trainer.py:1571
all_logps = per_token_logps[:, 1:].sum(-1)      # ← 求和，不是平均
```

实测同一批数据：

```text
logps/chosen: -425.5   logps/rejected: -321.0    ← rejected 更短，所以数值更不负
logps/chosen: -237.4   logps/rejected: -238.0    ← 长度接近时才对得上
```

**直接比较 chosen/rejected 的 logps 是错的**，它主要反映长度差。

**但 DPO 为什么还能工作？** 因为 `logits = (chosen - rejected) - (ref_chosen - ref_rejected)` 做了**两次相减**，
参考模型在同两条序列上算出同量级的长度偏差，**模型无关的那部分被抵消了**。

**残留的是"策略自身的长度偏好"** —— 这就是文献里"DPO 偏好更长输出"的来源。
而 TRL 0.21.0 的 `DPOConfig` **没有** `average_log_prob` 这个归一化开关
（它只作为 liger kernel 的内部参数硬编码为 `False`）。

### 8.4 诚实的结论

```text
- SFT → DPO 链路在 8GB 上跑得通，显存 2.76 GB allocated / 3.56 GB reserved
- 本次 DPO 没有产生可测量的偏好提升：rewards/accuracies 0.455，低于随机线
- 主要原因是训练量太小（约 15 次优化器更新），lr=5e-6 下 adapter 最大变化仅 2.04e-04
- 其次是两个方法论缺陷（参考模型错位、logps 长度偏差）会稀释信号
- 局限：仅 0.5B、仅 1000 条 UltraFeedback、单次运行、未做 held-out 偏好评测
  不能据此断言"DPO 无效"，只能说这个配置规模不足以产生效果
```

**最后一句很重要**：区分"方法无效"和"这次配置不足"。
一个诚实的实验记录不会把负面结果包装成正面，也不会把"规模不够"说成"方法不行"。

---

## 9. 显存分析：为什么 allocated 涨了但 reserved 降了

| | peak_allocated | peak_reserved | 步数 |
|---|---|---|---|
| SFT | 1.87 GB | **4.89 GB** | 250 |
| DPO | 2.76 GB | **3.56 GB** | 125 |

### `allocated` 涨 48% —— 符合预期

DPO 每步要做：

```text
1. 策略前向：chosen + rejected 两条序列（拼接成一个 batch）
2. 参考前向：同样两条序列，但 adapter 关闭
   → 多一次完整的前向
```

**多一次前向 + 双倍序列 → `allocated` 上升是必然的。**

### `reserved` 反而更低 —— 这是一个分析陷阱

**原因：`reserved` 随运行步数增长。**

```text
PyTorch caching allocator 的行为：
  需要显存 → 从池子里找 → 找不到就向驱动申请一大块
  释放张量 → 归还池子，不还驱动
        ↓
  步数越多 → 池子里的缓存块越多 → reserved 单调上升
```

SFT 跑了 250 步、DPO 只有 125 步 → **SFT 的 `reserved` 更高是步数差异造成的，
不是 SFT 更耗显存。**

**结论：跨实验比显存必须固定步数，否则结论是假的。**

> 这条已写进 `docs/8gb-memory-guide.md`。**这是一个很容易得出错误结论的地方** ——
> 如果不加分析地看表格，会得出"DPO 比 SFT 省显存"的相反结论。

---

## 本篇小结：DPO 脚本的设计要点

1. **命令行参数 > 配置值 > 默认。** `--adapter or cfg["start_from"]` 让正式实验可复现、
   临时试验不改配置。
2. **`is_trainable=True` 是必须的。** 默认 `False` 导致优化器为空，
   训练静默失效，而 `loss` / `grad_norm` 全部"正常"。
3. **`AdamW([])` 不报错** —— 这是 bug 能隐藏的根本原因。
4. **参考前向的 `disable_adapter()` 有副作用**，会把 `requires_grad` 打开。
   时序上它在建优化器**之后**，所以救不了空优化器。
5. **唯一可靠的验证是产物比对**（SHA256 / 逐张量 `max_abs_diff`）。
6. **防呆守卫只检查 `== 0`**，用 `raise` 不用 `assert`，且必须放在
   "adapter 已手动挂上"的那条路径上。
7. **`peft_config=None` + 手动挂 adapter** 是互斥的两条路，用 `elif` 保证。
8. **`ref_model=None` 时参考模型是底座，不是 SFT 模型** —— 这是方法论缺陷，
   要记录而不是隐藏。
9. **`beta` 同时影响 loss 和 reward 的数值**，跨实验比较时注意。
10. **`loss_type="sft"` 是一个现成的消融选项**，可以用来验证"偏好学习是否真的起作用"。
11. **`logps` 是求和，长度有偏**；两次相减抵消了模型无关的部分，但残留长度偏好。
12. **`reserved` 随步数漂移**，跨实验比较必须固定步数。
13. **区分"方法无效"和"配置规模不足"。** 本次是后者。
