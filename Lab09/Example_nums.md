Below is a **very clear, step-by-step numerical example** showing:

1. ✔ How **convolution** works on a grayscale 5×5 image using a 3×3 filter
2. ✔ How **MaxPooling** works
3. ✔ How **AveragePooling** works

Everything uses **small numbers (0–9)** so you can compute it by hand.

---

# ✅ **Part 1 — Convolution Example (5×5 image, 3×3 filter, stride=1, no padding)**

## 🔹 **Input Image (5×5, grayscale)**

```
I =
[ 1  2  3  0  1
  4  5  6  1  2
  7  8  9  2  3
  1  2  1  0  1
  2  3  2  1  0 ]
```

## 🔹 **3×3 Filter (kernel)**

A simple edge detector-like filter:

```
F =
[ 1  0 -1
  1  0 -1
  1  0 -1 ]
```

## 🔹 **Output size calculation**

Since stride = 1, padding = 0:

[
\text{Output size} = \frac{5 - 3}{1} + 1 = 3
]

So output = **3×3**.

---

# ⭐ **Compute Convolution Step-by-Step**

We slide the filter over the image.

---

## 👉 **Output pixel (0,0)**

Take the top-left 3×3 block:

```
[1 2 3
 4 5 6
 7 8 9]
```

Multiply elementwise by filter **F**:

```
1*1  +  2*0  +  3*(-1)  +
4*1  +  5*0  +  6*(-1)  +
7*1  +  8*0  +  9*(-1)
```

Compute:

```
1 + 0 - 3 +
4 + 0 - 6 +
7 + 0 - 9 = -6
```

So:

```
O[0,0] = -6
```

---

## 👉 **Output pixel (0,1)**

Next 3×3 block:

```
[2 3 0
 5 6 1
 8 9 2]
```

Multiply and sum:

```
2*1 + 3*0 + 0*(-1) +
5*1 + 6*0 + 1*(-1) +
8*1 + 9*0 + 2*(-1)

= 2 + 0 + 0 + 5 + 0 -1 + 8 + 0 -2 = 12
```

So:

```
O[0,1] = 12
```

---

## 👉 **Output pixel (0,2)**

```
[3 0 1
 6 1 2
 9 2 3]
```

Compute:

```
3*1 + 0*0 + 1*(-1) +
6*1 + 1*0 + 2*(-1) +
9*1 + 2*0 + 3*(-1)

= 3 + 0 -1 + 6 + 0 -2 + 9 + 0 -3 = 12
```

So:

```
O[0,2] = 12
```

---

### ⭐ After computing all positions:

## 🎉 **Final Convolution Output (3×3)**

```
[ -6   12   12
  -6   12   12
  -6   12   12 ]
```

This is what a CNN internally computes.

---

# ✅ **Part 2 — MaxPooling Example (2×2, stride=2)**

Take a simple 4×4 example:

```
A =
[ 1  3  2  1
  4  6  5  2
  0  1  3  4
  2  1  2  0 ]
```

Divide into **non-overlapping 2×2 blocks**:

### Block 1:

```
[1 3
 4 6] → max = 6
```

### Block 2:

```
[2 1
 5 2] → max = 5
```

### Block 3:

```
[0 1
 2 1] → max = 2
```

### Block 4:

```
[3 4
 2 0] → max = 4
```

## 🎉 **MaxPool output (2×2)**

```
[ 6  5
  2  4 ]
```

---

# ✅ **Part 3 — AveragePooling Example (2×2, stride=2)**

Using the same matrix **A**:

### Block 1:

```
[1 3
 4 6] → avg = (1+3+4+6) / 4 = 3.5
```

### Block 2:

```
[2 1
 5 2] → avg = 2.5
```

### Block 3:

```
[0 1
 2 1] → avg = 1.0
```

### Block 4:

```
[3 4
 2 0] → avg = 2.25
```

## 🎉 **AveragePool output (2×2)**

```
[ 3.5   2.5
  1.0   2.25 ]
```

---

# ✅ Summary (Very Important)

| Operation       | Input | Output | What Happens                   |
|-----------------|-------|--------|--------------------------------|
| **Convolution** | 5×5   | 3×3    | Multiply & sum using filters   |
| **MaxPooling**  | 4×4   | 2×2    | Take max of each 2×2 block     |
| **AvgPooling**  | 4×4   | 2×2    | Take average of each 2×2 block |

---
