# Find slope and equation from two points `(x1, y1)`, `(x2, y2)`

Given **two points**
$
(x_1, y_1) \quad \text{and} \quad (x_2, y_2)
$

---

## 🧩 1️⃣ Find the Slope (m)

The **slope** (or gradient) of the line passing through the two points is:

$$
m = \frac{y_2 - y_1}{x_2 - x_1}
$$

⚠️ Be careful: if $x_1 = x_2$, the slope is **undefined** (vertical line).

---

## 🧮 2️⃣ Equation of the Line

We can use the **point–slope form** of a line:

$$
y - y_1 = m(x - x_1)
$$

Rearranging to slope-intercept form $y = m x + c$:

$$
c = y_1 - m x_1
$$

So the **final equation** is:

$$
y = m x + c
$$

---

## 🧠 3️⃣ Example

Let’s use your example points:
$(x_1, y_1) = (0, 1), (x_2, y_2) = (1, 0)$

$$
m = \frac{0 - 1}{1 - 0} = -1
$$

$$
c = 1 - (-1)(0) = 1
$$

✅ So the equation is:

$$
y = -x + 1
$$

---

## 🧮 4️⃣ In Python

```python
def line_from_points(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


m, c = line_from_points(0, 1, 1, 0)
print(f"Slope = {m}, Equation: y = {m}x + {c}")
```

Output:

```
Slope = -1.0, Equation: y = -1.0x + 1.0
```

---

