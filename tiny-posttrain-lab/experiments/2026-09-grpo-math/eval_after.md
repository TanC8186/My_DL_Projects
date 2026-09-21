# Math evaluation

- tag: `grpo_after`
- model: `E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct`
- adapter: `experiments\2026-09-grpo-math\output\final`
- data: `data\math_eval.jsonl` (100 rows)
- decoding: greedy, max_new_tokens=256, seed=42
- VRAM: peak_allocated=0.94 GB, peak_reserved=1.0 GB

**Accuracy: 88.0/100 = 0.8800**

## Failures (12)

### gold=280 pred=270.0

> A book has 318 pages. You read 38. How many pages are left? End your response with the number only.

```text
270 pages are left.
```

### gold=513 pred=523.0

> Compute 27 * 19. End your response with the number only.

```text
523
```

### gold=308 pred=288.0

> Compute 14 * 22. End your response with the number only.

```text
288
```

### gold=340 pred=330.0

> A book has 398 pages. You read 58. How many pages are left? End your response with the number only.

```text
330
```

### gold=48 pred=56.0

> There are 4 bags with 12 apples each. How many apples in total? End your response with the number only.

```text
56
```

### gold=121 pred=110.0

> There are 11 bags with 11 apples each. How many apples in total? End your response with the number only.

```text
110
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

### gold=27 pred=28.0

> Compute 324 / 12. End your response with the number only.

```text
28
```

### gold=69 pred=79.0

> What is 25 + 44? End your response with the number only.

```text
79
```

### gold=229 pred=139.0

> A book has 290 pages. You read 61. How many pages are left? End your response with the number only.

```text
139页
```
