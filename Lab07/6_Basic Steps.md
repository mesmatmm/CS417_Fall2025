# 6 Basic steps to build a neural network using `Keras`, with practical explanations, tips, and examples.

---

## ⚙️ Basic Steps to Build a Neural Network

Here’s the **general workflow**:

1. **Import libraries**
2. **Prepare the data**
3. **Build the model**
4. **Compile the model**
5. **Train the model**
6. **Evaluate and make predictions**

---

## ⚙️ Step 1: **Import Libraries**

You start by importing TensorFlow and Keras.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### 🔍 What’s Happening:

* **TensorFlow** is the backend engine that performs all numerical computation.
* **Keras** is a high-level API that simplifies model building, training, and evaluation.
* **layers** is a module where you find layer types like `Dense`, `Conv2D`, `Dropout`, etc.

### 💡 Tips:

* Always use `tensorflow.keras` (not the standalone `keras`) to avoid version mismatches.
* You can check your TensorFlow version with:

  ```python
  print(tf.__version__)
  ```

---

## 🧮 Step 2: **Prepare the Data**

You can’t train a model without properly prepared data.

### Tasks Involved:

1. **Loading data** — from datasets, CSV files, or APIs.
2. **Splitting data** — into training, validation, and test sets.
3. **Normalizing/scaling** — helps faster convergence.
4. **Reshaping** — for image or sequence data.
5. **Encoding labels** — converting categories to numbers.

### 🧠 Example:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize (convert pixel range from [0, 255] to [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten 28x28 images into 1D vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
```

### 💡 Tips:

* For **images**, normalization is essential.
* For **categorical labels**, use `to_categorical()` for one-hot encoding if needed:

  ```python
  y_train = keras.utils.to_categorical(y_train, num_classes=10)
  ```

---

## 🧩 Step 3: **Build the Model**

You define the **architecture** — the number of layers, type of layers, and activation functions.

### Example (Sequential API):

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Components:

| Part              | Description                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------|
| **Input Layer**   | Defines input shape (number of features).                                                           |
| **Hidden Layers** | Process information with activations (e.g. ReLU).                                                   |
| **Output Layer**  | Gives final result; activation depends on task (`softmax` for classification, none for regression). |

### 💡 Tips:

* Start small; overfitting happens fast with too many neurons.
* For complex architectures (branching, shared layers), use the **Functional API**.

---

## ⚙️ Step 4: **Compile the Model**

You tell Keras how to train the model — what optimizer, loss, and metrics to use.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Explanation:

| Parameter     | Description                                                          |
|---------------|----------------------------------------------------------------------|
| **optimizer** | Controls *how* weights are updated (e.g. `adam`, `sgd`, `rmsprop`).  |
| **loss**      | Measures how far the model’s predictions are from the target values. |
| **metrics**   | Used to monitor performance (e.g. accuracy, MAE, etc.).              |

### 💡 Tips:

* For **classification**: use `categorical_crossentropy` or `sparse_categorical_crossentropy`.
* For **regression**: use `mse` (mean squared error).
* Experiment with learning rate:

  ```python
  keras.optimizers.Adam(learning_rate=0.001)
  ```

---

## 🏋️ Step 5: **Train (Fit) the Model**

This is where learning happens — the model adjusts its weights to minimize loss.

```python
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)
```

### Parameters:

| Parameter            | Meaning                                                         |
|----------------------|-----------------------------------------------------------------|
| **epochs**           | Number of passes over the full dataset.                         |
| **batch_size**       | Number of samples processed before weights are updated.         |
| **validation_split** | Fraction of training data used for validation.                  |
| **verbose**          | Controls how much info is printed during training (0, 1, or 2). |

### 💡 Tips:

* Use `EarlyStopping` to prevent overfitting:

  ```python
  callback = keras.callbacks.EarlyStopping(patience=3)
  ```
* Use `history.history` to visualize loss/accuracy over epochs.

---

## 🧪 Step 6: **Evaluate and Make Predictions**

After training, test the model on unseen data.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")
```

Make predictions:

```python
predictions = model.predict(x_test[:5])
print(predictions.argmax(axis=1))
```

### 💡 Tips:

* Use `model.evaluate()` for performance metrics.
* For regression:

  ```python
  predictions = model.predict(x_test)
  print(predictions[:5])
  ```
* You can **save and load** trained models:

  ```python
  model.save('my_model.h5')
  new_model = keras.models.load_model('my_model.h5')
  ```

---

## 🧠 Summary Table

| Step                  | Description                     | Typical Functions                     |
|-----------------------|---------------------------------|---------------------------------------|
| 1️⃣ Import Libraries  | Load TensorFlow & Keras modules | `import tensorflow as tf`             |
| 2️⃣ Prepare Data      | Load, clean, split, normalize   | `train_test_split`, `StandardScaler`  |
| 3️⃣ Build Model       | Define architecture             | `keras.Sequential`, `layers.Dense`    |
| 4️⃣ Compile Model     | Specify training configs        | `model.compile()`                     |
| 5️⃣ Train Model       | Fit model to data               | `model.fit()`                         |
| 6️⃣ Evaluate/ Predict | Test and use the model          | `model.evaluate()`, `model.predict()` |

---
