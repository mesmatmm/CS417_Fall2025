# Perceptron for the line through (1,0) and (0,1)

**Step 1 — line equation.**
The straight line through the points ((1,0)) and ((0,1)) is
$$
x_1 + x_2 = 1 .
$$
Bring to the form $(w\cdot x + b = 0)$:
$$
1\cdot x_1 + 1\cdot x_2 - 1 = 0.
$$

**Step 2 — choose inputs, weights and bias.**

* Inputs: $x = (x_1,x_2)$.
* Weights: $w = (1,,1)$.
* Bias: $b = -1$.

* So the perceptron computes $$z = w\cdot x + b = x_1 + x_2 - 1$$.

**Step 3 — activation function.**

Use a step (Heaviside) activation:
$$
\mathrm{output} =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
$$
With this choice, the perceptron outputs `1` for points satisfying $x_1 + x_2 \ge 1$ (on one side of the line) and `0`
for points with $x_1 + x_2 < 1$ (the other side).

**Which side is “inside our class”?**

**There is no general rule that “points above the line are inside” and “below are outside.”
It depends entirely on the perceptron’s weights and bias signs**.

You must check the **signs of the weights and bias**:

$$
\text{If } w_2 > 0 \Rightarrow \text{above the line ⇒ inside.}
$$
$$
\text{If } w_2 < 0 \Rightarrow \text{above the line ⇒ outside.}
$$

If you want the opposite region (points with $x_1 + x_2 < 1$ to be the class, either:

* flip the sign of weights and bias: (w=(-1,-1), b=+1); **or**
* keep (w=(1,1), b=-1) and interpret output `0` as inside.

**Quick examples**

* Point ((0,0)): $z=0+0-1=-1 <0 \Rightarrow$ output 0.
* Point ((0.6,0.6)): $z=1.2-1=0.2\ge0 \Rightarrow$ output 1.
* Point $(1,0)$ or $(0,1)$: $z=0\Rightarrow$ on the boundary (you may treat boundary as inside by using $\ge0$).

**Optional: perceptron as augmented input**

You can use an augmented input $\tilde x=(x_1,x_2,1)$ and weights $\tilde w=(1,1,-1)$. Then $z=\tilde w\cdot\tilde x$.

---

**Tiny Python demo**

```python
def step(z): return 1 if z >= 0 else 0


w = (1, 1)
b = -1


def perceptron(x):
    z = w[0] * x[0] + w[1] * x[1] + b
    return step(z)


tests = [(0, 0), (0.6, 0.6), (1, 0), (0, 1), (0.2, 0.7)]
for p in tests:
    print(p, perceptron(p))
```

That’s it — the perceptron with $w=(1,1), b=-1$ implements the decision boundary $x_1+x_2=1$. Flip signs or flip label
interpretation if you need the opposite side labeled as the positive class.
