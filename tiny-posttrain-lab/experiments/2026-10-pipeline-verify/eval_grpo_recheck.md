# Math evaluation

- tag: `eval_recheck`
- model: `E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct`
- adapter: `experiments/2026-09-grpo-math/output/final`
- data: `data\math_eval.jsonl` (100 rows)
- decoding: sample(t=0.7), max_new_tokens=256, seed=42
- VRAM: peak_allocated=0.94 GB, peak_reserved=1.0 GB

**Accuracy: 88.0/100 = 0.8800**

## Failures (12)

### gold=513 pred=523.0

> Compute 27 * 19. End your response with the number only.

```text
523
```

### gold=51 pred=41.0

> A pen costs 3 yuan. How much do 17 pens cost? End your response with the number only.

```text
41
```

### gold=340 pred=330.0

> A book has 398 pages. You read 58. How many pages are left? End your response with the number only.

```text
330页
```

### gold=48 pred=56.0

> There are 4 bags with 12 apples each. How many apples in total? End your response with the number only.

```text
56
```

### gold=121 pred=11.0

> There are 11 bags with 11 apples each. How many apples in total? End your response with the number only.

```text
11
```

### gold=73 pred=93.0

> A book has 115 pages. You read 42. How many pages are left? End your response with the number only.

```text
93
```

### gold=260 pred=250.0

> A book has 296 pages. You read 36. How many pages are left? End your response with the number only.

```text
250
```

### gold=129 pred=132.0

> Compute 3 * 43. End your response with the number only.

```text
132
```

### gold=624 pred=584.0

> Compute 24 * 26. End your response with the number only.

```text
584
```

### gold=27 pred=29.0

> Compute 324 / 12. End your response with the number only.

```text
29.0
```

### gold=69 pred=79.0

> What is 25 + 44? End your response with the number only.

```text
79
```

### gold=229 pred=130.0

> A book has 290 pages. You read 61. How many pages are left? End your response with the number only.

```text
130 pages are left.
```
