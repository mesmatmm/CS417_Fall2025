# 🧠 Normalization (Feature Scaling):

---

* It is common that the values of features in a dataset come in different scales.
* It is common to apply some type of **feature scaling** (also known as **normalization**) to the given data to make the
  scale of features "comparable".
* Two common ways for feature scaling:
    * **Standardization**
    * **Min-Max Scaling**

---

### َQ: Why Scaling is Important

- Neural networks perform better and converge faster when features are normalized.
- It prevents features with large numeric ranges (like 7.0 vs 0.2) from dominating others.
- Helps the optimizer (like Adam) adjust weights more evenly.

---

## 🧠 1️⃣ Standardization (Z-score Normalization)

### 📘 **Concept**

Standardization rescales features so that:

* Mean = 0
* Standard deviation = 1

It centers the data around **zero**, preserving the **shape** of the original distribution but normalizing its scale.

---

### 🧮 **Formula**

$$
z = \frac{x - \mu}{\sigma}
$$

where:

* $ x $ = original value
* $ \mu $ = mean of the feature
* $ \sigma $ = standard deviation of the feature

---

### 🧩 **Example**

Suppose we have:

| Original values |
|-----------------|
| 10              |
| 12              |
| 14              |
| 16              |
| 18              |

* Mean (μ) = 14
* Standard deviation (σ) = 3.16

Now:
$$
z = \frac{x - 14}{3.16}
$$

| x  | z-score |
|----|---------|
| 10 | -1.27   |
| 12 | -0.63   |
| 14 | 0.00    |
| 16 | 0.63    |
| 18 | 1.27    |

→ Now the data is centered at 0, with spread ≈ 1.

---

### 🧑‍💻 **Code Example**

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example data
X = np.array([[10], [12], [14], [16], [18]])

scaler = StandardScaler()  # Create a scaler object
X_scaled = scaler.fit_transform(
    X)  # Learn the mean and std of each feature, then Apply normalization using learned values

print("Original:\n", X.flatten())
print("Standardized:\n", X_scaled.flatten())
```

**Output:**

```
Original:
 [10 12 14 16 18]
Standardized:
 [-1.26 -0.63  0.00  0.63  1.26]
```

---

## 🧮 2️⃣ Min–Max Scaling (Normalization)

### 📘 **Concept**

Min–Max scaling rescales data to a fixed **range**, usually between **0 and 1** (or sometimes -1 to 1).
It preserves the shape of the original distribution but shifts and scales it.

---

### 🧮 **Formula**

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

where:

* $ x_{min} $ and $ x_{max} $ are the minimum and maximum values in the feature.

---

### 🧩 **Example**

Same data:

| x  | Min–Max scaled |
|----|----------------|
| 10 | 0.00           |
| 12 | 0.25           |
| 14 | 0.50           |
| 16 | 0.75           |
| 18 | 1.00           |

---

### 🧑‍💻 **Code Example**

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([[10], [12], [14], [16], [18]])

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

print("Original:\n", X.flatten())
print("Min–Max Scaled:\n", X_scaled.flatten())
```

**Output:**

```
Original:
 [10 12 14 16 18]
Min–Max Scaled:
 [0.   0.25 0.5  0.75 1.  ]
```

---

## ⚖️ 3️⃣ **Comparison: Standardization vs Min–Max Scaling**

| Feature                 | **Standardization (Z-score)**                                                            | **Min–Max Scaling**                                                                             |
|-------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Formula                 | (x − μ) / σ                                                                              | (x − min) / (max − min)                                                                         |
| Range                   | No fixed range (typically around −3 to +3)                                               | Fixed range (usually 0 to 1)                                                                    |
| Affected by outliers?   | Less sensitive                                                                           | Very sensitive                                                                                  |
| Keeps shape of data?    | Yes                                                                                      | Yes                                                                                             |
| Typical use case        | Algorithms assuming normal distribution (e.g. SVM, Logistic Regression, Neural Networks) | When all features must be in same fixed range (e.g. image data, distance-based models like KNN) |
| Common with Neural Nets | ✅ Yes (often preferred)                                                                  | ✅ Also common (esp. image pixels)                                                               |

---

## 🧭 4️⃣ **Which One Is Best?**

It depends on your problem:

### ✅ Use **Standardization** when:

* Your data has **outliers**.
* Features have **different distributions**.
* You use algorithms that assume normality (e.g., **SVM, Logistic Regression, Neural Networks**).
* You don’t need all values strictly between 0 and 1.

💡 *Most neural networks (especially with ReLU or tanh activations) work better with standardized inputs.*

---

### ✅ Use **Min–Max Scaling** when:

* You know the **feature range is bounded**, e.g., [0, 255] for image pixels.
* You use algorithms based on **distances** (e.g., KNN, K-Means).
* You want all features in the **same range**.

---

### 🧠 **In summary**

| When to Use                           | Recommended Method   |
|---------------------------------------|----------------------|
| Neural networks (general)             | **StandardScaler()** |
| Image pixels (0–255)                  | **MinMaxScaler()**   |
| Distance-based methods (KNN, K-Means) | **MinMaxScaler()**   |
| Data with outliers                    | **StandardScaler()** |

---
