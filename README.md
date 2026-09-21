# My_DL_Projects

大模型工程练习工作区（预训练 / 后训练实习向）。

## 当前项目

| 目录 | 方向 | 说明 |
|------|------|------|
| `tiny-posttrain-lab/` | 方向一：8G 可复现后训练实验库 | **主项目**，从 SFT 开始 |
| `references/trl/` | 第三方源码 | Hugging Face TRL，只读参考 |
| `qwen/` | 本地权重 | Qwen2.5-0.5B / 1.5B / 3B-Instruct |

## 开始动手

见 [`tiny-posttrain-lab/README_zh.md`](./tiny-posttrain-lab/README_zh.md) 与 [`tiny-posttrain-lab/docs/project-layout.md`](./tiny-posttrain-lab/docs/project-layout.md)。

```powershell
cd E:\My_DL_Projects\tiny-posttrain-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml
```

## Git

本工作区已 `git init`（branch: `master`）。  
尚未提交；`qwen/`、`references/` 默认已被 ignore。首次提交请只暂存 `tiny-posttrain-lab/` 等自有文件。
