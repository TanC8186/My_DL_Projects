# Math evaluation

- tag: `base`
- model: `E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct`
- adapter: none
- data: `data\math_eval.jsonl` (100 rows)
- decoding: greedy, max_new_tokens=256, seed=42
- VRAM: peak_allocated=0.94 GB, peak_reserved=0.99 GB

**Accuracy: 87.0/100 = 0.8700**

## Failures (13)

### gold=280 pred=270.0

> A book has 318 pages. You read 38. How many pages are left? End your response with the number only.

```text
270
```

### gold=513 pred=523.0

> Compute 27 * 19. End your response with the number only.

```text
523
```

### gold=65 pred=70.0

> There are 5 bags with 13 apples each. How many apples in total? End your response with the number only.

```text
70
```

### gold=369 pred=269.0

> A book has 399 pages. You read 30. How many pages are left? End your response with the number only.

```text
269
```

### gold=340 pred=230.0

> A book has 398 pages. You read 58. How many pages are left? End your response with the number only.

```text
230
```

### gold=121 pred=110.0

> There are 11 bags with 11 apples each. How many apples in total? End your response with the number only.

```text
110
```

### gold=73 pred=93.0

> A book has 115 pages. You read 42. How many pages are left? End your response with the number only.

```text
93
```

### gold=260 pred=150.0

> A book has 296 pages. You read 36. How many pages are left? End your response with the number only.

```text
150
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
29
```

### gold=69 pred=79.0

> What is 25 + 44? End your response with the number only.

```text
79
```

### gold=229 pred=139.0

> A book has 290 pages. You read 61. How many pages are left? End your response with the number only.

```text
139
```
