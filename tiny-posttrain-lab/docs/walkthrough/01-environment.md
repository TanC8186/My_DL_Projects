# 01 · 环境与工程配置

这一篇讲 5 个"非算法"文件。它们看起来最无聊，但**如果这一层不对，后面所有代码都跑不起来**。

| 文件 | 作用 |
|------|------|
| `requirements.txt` | 精确版本锁定 + 环境踩坑说明 |
| `pyproject.toml` | 包元数据 + 依赖范围 + 代码风格配置 |
| `.gitignore`（根 + 项目各一份） | 什么进版本库、什么不进 |
| `scripts/env.ps1` | 环境变量，解决 `USE_TF` 崩溃 |
| `scripts/run_sft.cmd` | 一键运行包装 |

---

## 1. `requirements.txt` —— 精确锁定

```text
# Local 8GB lab — install on Windows PowerShell
# Tested working combo (2026-09 baseline on RTX 4060 Laptop 8GB):
#   Base: conda env dl_env (Python 3.10) + torch 2.5.1+cu124
#   venv: python -m venv .venv --system-site-packages  (reuse local CUDA torch)
#
# Before training, set:
#   $env:USE_TF = "0"
#   $env:USE_FLAX = "0"
#   $env:HF_ENDPOINT = "https://hf-mirror.com"   # optional, CN network
#
# Do NOT install transformers 5.x / trl 1.x together on this machine —
# `from trl import SFTTrainer` can hard-crash (0xC0000005).
# Also avoid importing a broken TensorFlow inside dl_env (NumPy 2.x conflict).

transformers==4.55.2
datasets==3.6.0
accelerate==1.6.0
peft==0.17.1
trl==0.21.0
pyyaml>=6.0
pandas>=2.0.0
numpy>=1.26.0
tqdm>=4.66.0
```

### 逐段解释

**注释块本身是文档，不是废话。** 它记录了四件如果不写就必然忘记的事：

1. **基础环境**：venv 建在 conda `dl_env`（Python 3.10）之上，用 `--system-site-packages`
   复用已经装好的 CUDA torch。这是为了**避免重装 2.5 GB 的 torch+cu124**。
2. **必须设的环境变量**：`USE_TF=0` / `USE_FLAX=0`（原因见第 4 节）。
3. **不要装的组合**：`transformers 5.x` + `trl 1.x` 在这台机器上
   `from trl import SFTTrainer` 会**硬崩溃（0xC0000005，Windows 访问冲突）**。
   这不是 Python 异常，是进程级崩溃 —— 调试器都抓不到栈。所以必须锁版本。
4. **网络**：国内用 HF 镜像。

**为什么用 `==` 而不是 `>=`？**

因为这里的每一个版本都是**实测跑通过的组合**。`transformers` / `trl` / `peft` 三者
互相耦合很紧（`trl` 依赖 `transformers` 的内部 API），放宽版本等于把"能不能跑"
交给运气。这类项目的第一价值是**可复现**，所以锁死。

`pyyaml` / `pandas` / `numpy` / `tqdm` 用 `>=` 是因为它们不参与耦合，放宽更灵活。

**为什么有 `pandas`？** 代码里没直接用到。它是 `datasets` 的隐式依赖之一，
显式列出来是为了避免 `datasets` 升级后依赖变化导致的缺包。

**注释掉的 optional 块：**

```text
# Optional / platform-sensitive:
# bitsandbytes>=0.43.0   # 4-bit QLoRA; may fail on Windows — see README
# flash-attn             # optional; needs matching CUDA/torch
# tensorboard            # if report_to: ["tensorboard"]
```

- `bitsandbytes`：4-bit 量化（QLoRA）用。**Windows 上经常装不上**，所以默认关掉
  （配置里 `load_in_4bit: false`）。0.5B 模型 bf16 本来就塞得下，不需要量化。
- `flash-attn`：需要和 CUDA/torch 版本严格匹配才能编译。配置里用 `sdpa`
  （PyTorch 内置的 scaled dot-product attention）作为替代，**零安装成本**。
- `tensorboard`：只在想看 loss 曲线时装。

### 一个对比：`pyproject.toml` 里的依赖是另一套

```toml
dependencies = [
  "transformers>=4.46.0",
  "datasets>=2.20.0",
  "accelerate>=0.34.0",
  "peft>=0.12.0",
  "trl>=0.12.0",
  "pyyaml>=6.0",
]
```

**同一个项目为什么有两套依赖声明，而且约束宽严相反？**

| | `requirements.txt` | `pyproject.toml` |
|---|---|---|
| 用途 | **本机复现** | 包的分发声明 |
| 约束 | `==` 精确锁定 | `>=` 最低版本 |
| 读者 | 未来的你 | pip / 其他用户 |

这是标准做法：`pyproject.toml` 说"我需要什么能力"，`requirements.txt` 说
"我在这台机器上用什么版本验证过"。如果只留一个，要么别人装不上，要么你复现不了。

---

## 2. `pyproject.toml` —— 包元数据

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

**构建后端声明。** 告诉 pip「这个包用什么工具构建」。`setuptools` 是最通用的选择。

```toml
[project]
name = "tiny-posttrain-lab"
version = "0.1.0"
description = "8GB-friendly LLM post-training lab (SFT/DPO/GRPO) for internship portfolio"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "tiny-posttrain-lab contributors" }]
```

`requires-python = ">=3.10"` 不是随便写的 —— 代码里用了
`dict[str, Any]`、`str | Path`、`list[float] | None` 这些语法：

- `list[float]`（内置泛型）：Python 3.9+
- `str | Path`（PEP 604 联合类型）：Python 3.10+

所以 3.10 是真实下限。而且每个文件都有 `from __future__ import annotations`，
这让**注解变成字符串延迟求值**，从而在旧版本上也不会因为注解语法报错 ——
但 `requires-python` 仍然诚实地标了 3.10，因为运行时确实依赖。

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ruff>=0.6.0"]
```

**可选依赖组。** 装法：`pip install -e ".[dev]"`。
把测试/lint 工具和运行时依赖分开，是为了让"只想跑训练"的人不必装 pytest。

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

**`src/` 布局（src-layout）。** 这是关键设计：

```text
src/my_lab/          ← 包在这里
scripts/train_sft.py ← 脚本在这里，通过 sys.path 手动加入 src/
```

好处是**包和脚本物理分离**：`my_lab` 是可安装、可导入、可单测的库；
`scripts/` 是一次性入口。同时避免了"在项目根目录下 `import my_lab` 意外成功"
这种脏问题（那会掩盖安装错误）。

因为用了 src-layout 且没 `pip install -e .`，脚本里才有这段：

```python
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
```

- `parents[1]`：从 `scripts/train_sft.py` 往上两级到项目根
- `sys.path.insert(0, ...)`：把 `src/` 加到导入搜索路径最前面
- `# noqa: E402`：告诉 linter「我知道 import 不在文件顶部，这是故意的」
  （E402 = module level import not at top of file）

**这样设计的代价**是每个脚本都要重复这 4 行；**收益**是「不需要安装就能直接跑」。

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

`ruff` 是极快的 Python linter。`line-length = 100` 比 PEP8 默认的 79 宽 ——
对深度学习代码来说，长表达式（比如一长串 `from_pretrained(..., ...)` 参数）
拆成 79 字符反而更难读。

---

## 3. `.gitignore` —— 两份，各有分工

### 根 `.gitignore`

```gitignore
# Root ignore for the monorepo workspace
# Keep third-party clones out of *our* history if you prefer; comment out
# the next line if you intentionally want to vendor references.

references/          # ← HF TRL 的源码克隆（几百 MB），不进版本库
*.log
.venv/
venv/
__pycache__/
.pytest_cache/

# IDE / editor
.idea/               # ← PyCharm 配置，不该进库
.vscode/

# Never commit large model weights (qwen/ may live here locally)
qwen/                # ← 本地模型权重（0.5B+1.5B+3B ≈ 10 GB）
*.safetensors
*.bin
*.pt

# Lab outputs (ignore everything in output/ except the tiny raw metrics file)
tiny-posttrain-lab/experiments/**/output/**
!tiny-posttrain-lab/experiments/**/output/train_metrics.txt
```

### 项目 `.gitignore`

```gitignore
# Python
.venv/
venv/
__pycache__/
*.py[cod]           # ← 匹配 .pyc/.pyo/.pyd（字符集语法）
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Experiment outputs
experiments/**/output/**
!experiments/**/output/train_metrics.txt
experiments/**/runs/
*.safetensors
*.bin
*.pt
*.ckpt
wandb/
lightning_logs/
tensorboard/

# Data blobs
data/**/*.parquet
data/**/*.bin
!data/**/*.jsonl     # ← 但 jsonl 小样例要保留
!data/**/*.md
!data/**/*.csv

# Secrets / local
.env
*.log

# OS
Thumbs.db
.DS_Store
```

### 这里有三个值得讲透的知识点

#### 知识点 1：`!` 否定规则的**致命限制**

原始写法是：

```gitignore
experiments/**/output/      # 注意结尾的斜杠
!experiments/**/output/train_metrics.txt
```

**这个否定不生效。** 原因是 git 的官方规则：

> It is not possible to re-include a file if a parent directory of that file is excluded.

`output/` 结尾带斜杠 = **排除这个目录本身**。git 遇到被排除的目录会**直接跳过，不进去看**，
所以里面的 `!` 规则根本没机会被求值。

正确写法是把「排除目录」改成「排除目录里的内容」：

```gitignore
experiments/**/output/**                          # 排除 output/ 下的所有内容
!experiments/**/output/train_metrics.txt          # 现在可以豁免单个文件
```

`output/**` 只匹配目录**内部**，`output` 目录本身不被排除，git 才会进去遍历并应用否定规则。

**验证方法**（不要靠推理，要实测）：

```powershell
git check-ignore -v path/to/train_metrics.txt
```

- 输出以 `!` 开头 → 被否定规则匹配，**不会**被忽略（会被提交）✅
- 输出普通规则 → 仍被忽略 ❌

本项目实测结果：

```text
tiny-posttrain-lab/.gitignore:14:!experiments/**/output/train_metrics.txt   ← 豁免成功
tiny-posttrain-lab/.gitignore:12:experiments/**/output/**                   ← 权重仍被忽略
```

#### 知识点 2：**为什么两层 `.gitignore` 都要改**

根 `.gitignore` 里也有一条 `tiny-posttrain-lab/experiments/**/output/**`。
一开始我只改了项目内那份，结果 `train_metrics.txt` **仍然被忽略** ——
因为根规则先命中了。

**规则是叠加的**，不是就近覆盖。调试 `gitignore` 问题的第一步永远是
`git check-ignore -v <path>`，它会明确告诉你**是哪一行的哪条规则**在生效。

#### 知识点 3：**为什么偏偏要豁免 `train_metrics.txt`**

它是唯一的例外。`output/` 里有：

| 文件 | 大小 | 要不要提交 |
|------|------|-----------|
| `adapter_model.safetensors` | 8.5 MB | ❌ 可再生 |
| `tokenizer.json` | 11 MB | ❌ 从底座复制来的 |
| `optimizer.pt` | 17 MB × 4 | ❌ 只有续训才需要 |
| `train_metrics.txt` | **0.2 KB** | ✅ **这是唯一的原始事实记录** |

`notes.md` 是人写的解读（可能抄错），`train_metrics.txt` 是程序写出的原始数字。
**解读可以被质疑，原始数据不行。** 所以 0.2 KB 的文件值得进版本库。

---

## 4. `scripts/env.ps1` —— 环境变量

```powershell
param([switch]$Offline)

$env:USE_TF = "0"
$env:USE_FLAX = "0"
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if ($Offline) { $env:HF_HUB_OFFLINE = "1" }
```

### 为什么需要 `USE_TF=0`（这是本机最坑的一条）

完整因果链：

```text
1. venv 用 --system-site-packages 创建
        ↓
   为了复用 conda dl_env 里已装好的 torch 2.5.1+cu124（省 2.5 GB 下载）

2. 副作用：conda dl_env 里的 TensorFlow 也暴露出来了
        ↓
   那个 TF 是**用 NumPy 1.x 编译的**，而当前 NumPy 是 2.2.6

3. 平时没事，因为 transformers 不会主动 import TF
        ↓
   但 transformers 的 image_transforms.py 里有一句 `import tensorflow as tf`
   —— 这是**惰性导入（lazy import）**，只在被访问时才执行

4. peft → transformers → modeling_utils → loss.loss_utils → loss_d_fine
   → loss_for_object_detection → image_transforms → import tensorflow
        ↓
   导入链走到这里，把那个坏掉的 TF 拉起来

5. 结果：AttributeError: _ARRAY_API not found → ImportError: numpy.core._multiarray_umath failed
        ↓
   `import peft` 直接崩溃，报错栈从 peft 一路钻到 tensorflow
```

**`USE_TF=0` 的作用**：告诉 transformers「别去找 TensorFlow」，
它的惰性导入机制会直接跳过，返回一个 mock 对象。同理 `USE_FLAX=0` 跳过 JAX。

**为什么不用更干净的方案？** 比如重建一个不带 `--system-site-packages` 的 venv。
可以，但要重装 torch（2.5 GB）+ CUDA 运行时。**这是显存 8G 之外的另一个资源约束：磁盘和时间。**
所以选择了"用一个环境变量绕开"，并把它写进文档和脚本，而不是假装没这回事。

### `HF_ENDPOINT` 与 `HF_HUB_OFFLINE`

```powershell
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if ($Offline) { $env:HF_HUB_OFFLINE = "1" }
```

- `HF_ENDPOINT`：国内访问 huggingface.co 常超时，镜像站可用。
  **只在未设置时设置**（`if (-not ...)`），这样用户自己指定的值不会被覆盖。
- `HF_HUB_OFFLINE=1`：**只在 `-Offline` 时设置**。这个开关更狠 ——
  它禁止一切网络请求，只用本地缓存。

**为什么需要 `-Offline` 这个开关？** 因为踩过一个具体的坑：

```text
数据集 yahma/alpaca-cleaned 已经完整缓存在本地了，
但训练启动时仍然卡在联网检查 README.md，重试 5 次后超时失败。
```

`datasets` 库即使命中缓存，也会发 HEAD 请求确认远端版本。离线模式下它会
直接使用缓存并打印：

```text
Using the latest cached version of the dataset since yahma/alpaca-cleaned
couldn't be found on the Hugging Face Hub (offline mode is enabled).
```

### 为什么用 `param([switch]$Offline)` 和 dot-source

```powershell
. .\scripts\env.ps1            # 注意开头的点号
```

**dot-source（点号 + 空格 + 脚本路径）** 在当前 PowerShell 会话中执行脚本，
所以 `$env:XXX` 会留在当前会话。如果直接 `.\scripts\env.ps1`，脚本会在**子作用域**
执行，环境变量设完就丢。

`param()` 必须放在脚本**所有语句之前**（注释之前可以有注释）。所以这里
把一大段注释放在 `param()` 上面 —— 这是合法的，但 `param()` 之后不能再有
可执行语句之外的顶层语句混在参数声明前面。

---

## 5. `scripts/run_sft.cmd` —— 一键包装

```batch
@echo off
REM Run SFT baseline with env workarounds for this machine.
set USE_TF=0
set USE_FLAX=0
set HF_ENDPOINT=https://hf-mirror.com
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\train_sft.py" --config "configs\sft_qwen05b_8g.yaml"
```

逐行：

| 行 | 作用 |
|---|---|
| `@echo off` | 不回显每条命令（否则输出很乱） |
| `set XXX=0` | cmd 里设环境变量（对比 PowerShell 的 `$env:XXX`） |
| `cd /d "%~dp0.."` | `%~dp0` = 脚本所在目录（含结尾反斜杠），`..` 回到项目根；`/d` 允许跨盘符切换 |
| 最后一行 | 用**显式路径**调用 venv 里的 python，不依赖 `PATH` 或激活状态 |

**为什么用 `%~dp0` 而不是相对路径？** 因为 cmd 的当前目录是**调用者所在目录**，
不是脚本目录。用 `%~dp0` 保证无论从哪里调用都能正确定位项目根。

**为什么这是 `.cmd` 而不是 `.ps1`？** 双击即可运行，不需要改 PowerShell 执行策略。
它和 `env.ps1` 是**功能重叠的两种入口**：`env.ps1` 用于交互式会话（dot-source 后
手动跑各种命令），`run_sft.cmd` 用于"只想跑一下 SFT"。

> **小提示**：这个 `.cmd` 写死了 `sft_qwen05b_8g.yaml`。要跑 DPO/GRPO，
> 用 PowerShell 加 `env.ps1`，或自己复制改一个 `.cmd`。

---

## 本篇小结：环境层的设计原则

1. **精确锁定 + 注释原因。** `==` 保证可复现；注释解释每个非显然的决定。
2. **两套依赖声明分工明确。** `pyproject.toml` 说能力需求，`requirements.txt` 说验证过的版本。
3. **`src/` 布局隔离库与脚本。** 库可单测、可安装；脚本是一次性入口。
4. **环境变量而不是重建环境。** 在磁盘/时间受限时，用 `USE_TF=0` 绕过依赖冲突，
   并把这件事写进文档（而不是留下一个"只有我知道的魔法命令"）。
5. **gitignore 用实测验证，不靠推理。** `git check-ignore -v` 是唯一可信的判据。
6. **原始数据进版本库，可再生产物不进。** 0.2 KB 的 `train_metrics.txt` 比 8.5 MB 的
   adapter 更值得保存。
