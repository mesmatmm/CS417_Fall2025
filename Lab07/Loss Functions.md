# 📘 **Common Loss Functions in Neural Networks**

This is a **clear beginner-friendly explanation** followed by a **comprehensive table** that summarizes
the **most common loss functions** used in neural networks.

---

## 🧠 **Definition of a Loss Function**

A **loss function** (also called a **cost function** or **objective function**) measures **how far the model’s
predictions are from the actual target values**.

It gives the neural network a **numerical value** that represents the *error* between predicted output and true output.

---

## 🎯 **Purpose of a Loss Function**

* It **guides learning** — the smaller the loss, the better the model’s predictions.
* During training, **optimization algorithms** like *gradient descent* adjust model weights to **minimize the loss**.
* Without it, the model wouldn’t know *how well or badly* it’s performing.

---

## 🔍 **Quick Reference — Common Beginner Choices**

| Task Type                         | Recommended Loss                     | Keras Name                          |
|-----------------------------------|--------------------------------------|-------------------------------------|
| Regression (continuous output)    | **MSE** or **MAE**                   | `'mse'` or `'mae'`                  |
| Binary classification (2 classes) | **Binary Cross-Entropy**             | `'binary_crossentropy'`             |
| Multi-class classification        | **Categorical Cross-Entropy**        | `'categorical_crossentropy'`        |
| Multi-class (integer labels)      | **Sparse Categorical Cross-Entropy** | `'sparse_categorical_crossentropy'` |
| Imbalanced classification         | **Focal Loss**                       | via `tensorflow_addons`             |
| Image segmentation                | **Dice Loss** (custom)               | custom                              |
| Count prediction                  | **Poisson Loss**                     | `'poisson'`                         |

---

## 🧩 **Beginner Tips**

* For **regression** → use **MSE** or **MAE**
* For **binary classification** → use **Binary Cross-Entropy**
* For **multi-class classification** → use **Categorical Cross-Entropy** (with softmax)
* For **imbalanced data** → consider **Focal Loss**
* For **embedding or similarity tasks** → use **Cosine or Triplet Loss**

---

## 🧠 **Most Common Loss Functions in Neural Networks**

* **Mean Squared Error (MSE)**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$
    * Keras name: `'mse'` or `'mean_squared_error'` (or `tf.keras.losses.MeanSquaredError()`)
    * Used for: Regression (continuous targets)
    * Notes: Penalizes large errors strongly (square). Common default for regression.

* **Root Mean Squared Error (RMSE)**

    * Equation: $L=\sqrt{\dfrac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}$
    * Keras name: Not built-in as single loss — implement as `tf.sqrt(mse)` or a custom loss wrapper.
    * Used for: Regression (interpretable in target units)
    * Notes: Same as MSE but interpretable in the same units as $y$.

* **Mean Absolute Error (MAE)**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|$
    * Keras name: `'mae'` or `'mean_absolute_error'` (or `tf.keras.losses.MeanAbsoluteError()`)
    * Used for: Regression
    * Notes: Less sensitive to outliers than MSE; gradients are constant (not squared).

* **Mean Absolute Percentage Error (MAPE)**

    * Equation: $L=\dfrac{100}{n}\sum_{i=1}^{n}\left|\dfrac{y_i-\hat y_i}{y_i}\right|$
    * Keras name: `'mape'` or `'mean_absolute_percentage_error'`
    * Used for: Regression (when relative error matters)
    * Notes: Expresses error as percentage; unstable if true $y_i$ near 0.

* **Huber Loss**

    * Equation:
      $$
      L=\begin{cases}
      \dfrac{1}{2}(y-\hat y)^2 & \text{if }|y-\hat y|\le\delta\
      \delta\big(|y-\hat y|-\tfrac{1}{2}\delta\big) & \text{if }|y-\hat y|>\delta
      \end{cases}
      $$
    * Keras name: `tf.keras.losses.Huber(delta=...)` (alias `'huber'` in some APIs)
    * Used for: Regression (robust to outliers)
    * Notes: Quadratic for small errors, linear for large errors — combines MSE and MAE benefits.

* **Log-Cosh Loss**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}\log\big(\cosh(\hat y_i-y_i)\big)$
    * Keras name: `'log_cosh'` (or `tf.keras.losses.LogCosh()`)
    * Used for: Regression
    * Notes: Very similar to MSE near zero but less impacted by large outliers due to log(cosh) behavior.

* **Binary Cross-Entropy (a.k.a. Log Loss)**

    * Equation: $L=-\dfrac{1}{n}\sum_{i=1}^{n}\big[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\big]$
    * Keras name: `'binary_crossentropy'` or `tf.keras.losses.BinaryCrossentropy()`
    * Used for: Binary classification (use with `sigmoid` output)
    * Notes: Measures difference between true labels (0/1) and predicted probabilities.

* **Categorical Cross-Entropy**

    * Equation: $L=-\dfrac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c}\log(\hat y_{i,c})$
    * Keras name: `'categorical_crossentropy'` or `tf.keras.losses.CategoricalCrossentropy()`
    * Used for: Multi-class classification **with one-hot encoded labels** (use with `softmax` output)
    * Notes: Sum over classes; penalizes incorrect probability mass.

* **Sparse Categorical Cross-Entropy**

    * Equation: Same as categorical CE but labels $y_i$ are integer class indices (not one-hot).
    * Keras name: `'sparse_categorical_crossentropy'` or `tf.keras.losses.SparseCategoricalCrossentropy()`
    * Used for: Multi-class classification **with integer labels**
    * Notes: Equivalent loss but more memory-efficient when labels are integers.

* **Kullback–Leibler (KL) Divergence**

    * Equation: $L=\sum_{c=1}^{C} y_c \log!\left(\dfrac{y_c}{\hat y_c}\right)$ (often averaged over samples)
    * Keras name: `'kullback_leibler_divergence'` or `'kld'` (or `tf.keras.losses.KLDivergence()`)
    * Used for: Probability/distribution matching (e.g., VAEs, distillation, probabilistic outputs)
    * Notes: Requires valid probability distributions (non-negative, sum to 1). Not symmetric.

* **Hinge Loss**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}\max(0,1 - y_i\hat y_i)$ where $y_i\in{-1,+1}$
    * Keras name: `'hinge'` or `tf.keras.losses.Hinge()`
    * Used for: Binary classification in SVM-style models (labels −1/+1)
    * Notes: Encourages a margin between classes; not commonly used with probability outputs.

* **Squared Hinge Loss**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}\big(\max(0,1 - y_i\hat y_i)\big)^2$
    * Keras name: `'squared_hinge'` or `tf.keras.losses.SquaredHinge()`
    * Used for: Binary classification (SVM-style)
    * Notes: Smoother and penalizes large margin violations more than plain hinge.

* **Poisson Loss**

    * Equation: $L=\dfrac{1}{n}\sum_{i=1}^{n}\big(\hat y_i - y_i\log(\hat y_i)\big)$  (assuming $\hat y_i>0$)
    * Keras name: `'poisson'` or `tf.keras.losses.Poisson()`
    * Used for: Count regression (events per interval)
    * Notes: Assumes Poisson-distributed targets; predictions must be positive.

* **Cosine Similarity Loss**

    * Equation (similarity): $\text{cos}(\hat y,y)=\dfrac{\hat y\cdot y}{|\hat y||y|}$. As a loss one common form
      is $L = 1 - \dfrac{\hat y\cdot y}{|\hat y||y|}$ (Keras uses negative
      similarity: $L=-\dfrac{\hat y\cdot y}{|\hat y||y|}$)
    * Keras name: `'cosine_similarity'` (note: Keras returns negative cosine similarity).
    * Used for: Embedding/similarity tasks, metric learning, NLP (when direction matters)
    * Notes: Measures vector orientation similarity; scale of vectors ignored.

* **Triplet Loss**

    * Equation (per triplet): $L=\max(0,d(a,p)-d(a,n)+\text{margin})$ where $d(\cdot,\cdot)$ is distance.
    * Keras name: No single built-in; TensorFlow provides `tf.keras.losses.TripletSemiHardLoss()` or implement custom.
    * Used for: Metric learning / face recognition / embeddings
    * Notes: Trains anchor-positive pairs to be closer than anchor-negative pairs by margin.

* **Dice Loss (for segmentation)**

    * Equation (soft-Dice version): $L=1 - \dfrac{2\sum_i \hat y_i y_i}{\sum_i \hat y_i + \sum_i y_i}$
    * Keras name: Not built-in — implement as custom loss.
    * Used for: Image segmentation (especially when classes are imbalanced)
    * Notes: Measures overlap (like F1); often combined with cross-entropy.

* **Focal Loss**

    * Equation (binary focal example): $L = -\alpha (1-\hat y)^\gamma \log(\hat y)$ for positive class (and analogously
      for negative).
    * Keras name: Not in core Keras — available in `tensorflow_addons` as `tfa.losses.SigmoidFocalCrossEntropy()` (or
      implement custom).
    * Used for: Highly imbalanced classification (object detection, rare classes)
    * Notes: Down-weights easy examples and focuses training on hard/misclassified examples;
      hyperparameters $(\alpha,\gamma)$.

---