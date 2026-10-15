# Serving 精读提纲（Part 2 用，现在只需浏览）

在 `E:\My_DL_Projects\references\` 下只读，不要改第三方代码。

## 推荐顺序

1. **llama.cpp**（8G 最先能跑通）
   - `README.md`：量化、CPU/GPU offload
   - `examples/`：server / main 的启动方式
   - 概念：GGUF、`n_gpu_layers`、KV cache 在端侧的表现

2. **vLLM**（工业 RL rollout 标配）
   - `vllm/` 包内搜 attention / scheduler / cache
   - 概念：PagedAttention、continuous batching、prefix caching
   - 和后训练的联系：verl/OpenRLHF 用 vLLM 做 rollout，再和训练引擎同步权重

3. **SGLang**（国内厂认知度高）
   - `python/sglang`（路径以克隆结果为准）
   - 概念：RadixAttention、结构化生成、多轮复用前缀

## 面试常问

- TTFT 和 tokens/s 分别衡量什么？
- 为什么 serving 和 training 常用两套引擎？
- PagedAttention 解决什么问题？
- 8G 卡上选 llama.cpp 而不是完整 vLLM 集群的理由是什么？

## 与 tiny-posttrain-lab 的衔接

写 SFT/GRPO 笔记时加一句：

> 推理/rollout 若走 vLLM，可降低 RL 中生成阶段成本；本地 8G 用 llama.cpp 做同模型的 serving 对照。

不必现在实测，**先把后训练主线做完**。
