# General formulas for tensor shapes & parameter counts in CNNs (clear + concrete)

Below are the standard, general equations you can use to compute **output spatial sizes, number of channels, and number
of parameters** for convolution, pooling, flatten, and dense layers. After the formulas I give **two numeric examples**
that prove the formulas.

---

## Notation

* `H, W` — input height and width
* `C_in` — number of input channels (depth)
* `K_h, K_w` — kernel (filter) height and width
* `C_out` — number of filters (output channels)
* `S_h, S_w` — stride in height and width (if stride is scalar `S` then `S_h = S_w = S`)
* `P_h, P_w` — padding (number of zero rows/cols added on each *side*) in height and width
* `⌊x⌋` — floor, `⌈x⌉` — ceil
* `use_bias` — 1 if layer uses bias term, 0 if not

> Many frameworks use either `padding='valid'` (no padding, `P=0`) or `padding='same'` (special rule). Below I give the
**general explicit formula** using `P`, and also the usual shortcut for `same`.

---

## 1) Convolution (Conv2D) — output shape

### General formula (explicit padding):

```
H_out = floor((H + 2*P_h - K_h) / S_h) + 1
W_out = floor((W + 2*P_w - K_w) / S_w) + 1
Channels_out = C_out
=> output shape = (H_out, W_out, C_out)
```

* If `padding='valid'` → `P_h = P_w = 0`.
* If `padding='same'` (common TF convention) and `S_h = S_w = 1` → `H_out = H`, `W_out = W`.
* For general `padding='same'` with stride > 1, frameworks commonly produce `H_out = ceil(H / S_h)` and
  `W_out = ceil(W / S_w)` (you can compute `P` precisely if needed).

### Number of parameters (weights + biases):

```
Params_conv = (K_h * K_w * C_in) * C_out  +  (use_bias ? C_out : 0)
```

---

## 2) Pooling (MaxPool / AvgPool)

Pooling uses kernel `K_h x K_w`, stride `S_h x S_w` and padding `P_h x P_w`:

```
H_out_pool = floor((H + 2*P_h - K_h) / S_h) + 1
W_out_pool = floor((W + 2*P_w - K_w) / S_w) + 1
Channels_out_pool = C_in   # pooling does not change channels
=> output shape = (H_out_pool, W_out_pool, C_in)
```

* Common case: `K = (2,2)`, `S = (2,2)`, `P = 0` → halves H and W (floor halving).

Pooling has **no learned parameters**.

---

## 3) Strided convolution vs pooling

* Strided conv uses same conv formula with `S>1` and changes channels to `C_out`.
* Pooling reduces spatial dims but keeps same channel count.

---

## 4) Flatten

If input to Flatten has shape `(H_f, W_f, C_f)`:

```
Flatten output vector length = H_f * W_f * C_f
```

---

## 5) Dense (Fully connected) layer

If Dense maps from `N_in` units → `N_out` units:

```
Params_dense = N_in * N_out + (use_bias ? N_out : 0)
Output shape = (N_out,)
```

---

## 6) BatchNormalization (per-channel)

For Conv feature maps `(H, W, C)` BN usually has **2 or 4 learned parameters per channel** (scale & shift gamma,beta;
running mean/var are buffers not counted as training params in some contexts):

```
Params_BN ≈ 2 * C  (gamma and beta)    # plus non-trainable running mean/var
Output shape unchanged: (H, W, C)
```

---

# Numerical Example 1 — *conv with VALID padding and stride > 1*

**Given:**

* Input image: `H=28, W=28, C_in=3` (e.g., 28×28 RGB)
* Conv layer: `K=3×3`, `C_out=32` filters, `stride = (2,2)`, `padding='valid'` (so `P_h=P_w=0`), `use_bias = 1`.

**Compute output spatial dims:**

```
H_out = floor((28 + 2*0 - 3) / 2) + 1 = floor(25 / 2) + 1 = 12 + 1 = 13
W_out = same = 13
Channels_out = C_out = 32
=> output shape = (13, 13, 32)
```

**Compute params:**

```
Params_conv = (K_h * K_w * C_in) * C_out + C_out
            = (3 * 3 * 3) * 32 + 32
            = (27) * 32 + 32
            = 864 + 32 = 896
```

So the conv layer has **896 parameters** and outputs a tensor of shape **(13, 13, 32)**.

**Check intuitively:** each of the 32 filters has `3×3×3 = 27` weights plus 1 bias → `28` parameters per filter →
`28 * 32 = 896`. Good.

---

# Numerical Example 2 — *conv with SAME padding and stride = 1, then pooling*

**Given pipeline:**

1. Input: `H=32, W=32, C_in=1` (grayscale 32×32)
2. Conv layer A: `K=3×3`, `C_out=16`, `stride=(1,1)`, `padding='same'`, `use_bias=1`.
3. BatchNorm + ReLU (shape stays same).
4. MaxPool: `K_pool = 2×2`, `stride = 2` (default often equal to pool size), `padding = 0`.

**Step 1 — Conv A output shape**
For `same` and `stride=1` the output spatial dims equal input:

```
H_out = H = 32
W_out = W = 32
Channels_out = C_out = 16
=> (32, 32, 16)
```

**Step 1 — Conv A params:**

```
Params_convA = (3*3*1)*16 + 16 = 9*16 + 16 = 144 + 16 = 160
```

So conv A has **160 parameters**.

**Step 2 — BatchNorm**
Output shape unchanged `(32, 32, 16)`. BN trainable params ≈ `2 * C = 32` (gamma & beta), so add 32 trainable params if
counting BN.

**Step 3 — MaxPool 2×2, stride 2**
Apply pooling formula (P=0):

```
H_pool = floor((32 + 0 - 2) / 2) + 1 = floor(30 / 2) + 1 = 15 + 1 = 16
W_pool = 16
Channels remain = 16
=> output shape after pool = (16, 16, 16)
```

**Flatten then Dense example:**

* Flatten length = `16 * 16 * 16 = 4096`
* Dense to 128 units (with bias):

```
Params_dense = 4096 * 128 + 128 = 524,288 + 128 = 524,416
Output shape = (128,)
```

**Summary of shapes for Example 2 pipeline:**

```
Input: (32, 32, 1)
Conv(3x3, 16, same, s=1) -> (32, 32, 16)    (params 160)
BatchNorm -> (32, 32, 16)                   (params ~32)
ReLU -> (32, 32, 16)
MaxPool(2x2, s=2) -> (16, 16, 16)
Flatten -> (4096,)
Dense(128) -> (128)                         (params 524,416)
Dense(10) -> (10)                           (params 128*10 + 10 = 1,290)
```

Everything above follows the general formulas and the arithmetic checks out.

---

# Quick reference cheat-sheet (copyable)

### Conv2D output size (explicit padding)

```
H_out = floor((H + 2*P_h - K_h) / S_h) + 1
W_out = floor((W + 2*P_w - K_w) / S_w) + 1
Channels_out = C_out
Params = (K_h*K_w*C_in) * C_out + (use_bias ? C_out : 0)
```

### Pooling output

```
H_out = floor((H + 2*P_h - K_h) / S_h) + 1
W_out = floor((W + 2*P_w - K_w) / S_w) + 1
Channels_out = same as input
```

### Flatten

```
Length = H * W * C
```

### Dense

```
Params = N_in * N_out + (use_bias ? N_out : 0)
Output = (N_out,)
```

---

