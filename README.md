# CS417_Fall2025

**CS417 (Neural Networks) — Fall 2025**

This repository holds the lab material for the CS417 Neural Networks course. It takes you from Python
and NumPy fundamentals, through the perceptron and multilayer networks trained with backpropagation,
up to building and training real models with Keras/TensorFlow and finally Convolutional Neural
Networks (CNNs). Each lab folder mixes hands-on code (Python scripts and Jupyter notebooks) with short
explanatory notes, and the lecture slides plus the shared datasets used across the labs are kept in
their own folders.

---

## Table of Contents

- [Labs](#labs)
  - [Lab01 — Python Basics](#lab01--python-basics)
  - [Lab02 — NumPy](#lab02--numpy)
  - [Lab03 — The Perceptron & MLP for Logic Functions](#lab03--the-perceptron--mlp-for-logic-functions)
  - [Lab04 — More Boolean Functions with MLPs](#lab04--more-boolean-functions-with-mlps)
  - [Lab05 — OOP, Lines & Perceptron Learning](#lab05--oop-lines--perceptron-learning)
  - [Lab06 — Backpropagation](#lab06--backpropagation)
  - [Lab07 — Deep Learning with Keras](#lab07--deep-learning-with-keras)
  - [Lab08 — Practical Models & Intro to CNNs](#lab08--practical-models--intro-to-cnns)
  - [Lab09 — CNN Shapes, Parameters & Pooling](#lab09--cnn-shapes-parameters--pooling)
- [Lectures](#lectures)
- [Project](#project)
- [Data](#data)
- [Course Staff](#course-staff)

---

## Labs

### Lab01 — Python Basics

- Variables, printing and f-string formatting.
- Built-in data types (`int`, `float`, `str`, `bool`, `complex`) and type checking with `type()`.
- User input with `input()` and type casting between types.
- Basic math operators, conditional statements (`if` / `elif` / `else`) and logical operators.
- `for` and `while` loops.
- Tasks: pairs of numbers summing to a target, and counting element frequencies with a dictionary.

### Lab02 — NumPy

- Why NumPy: fast arrays, matrices and vectorized math.
- Creating arrays: zeros, ones, identity, random matrices and `repeat`.
- Shallow copy vs. deep copy of arrays.
- Basic mathematical and element-wise operations.
- Linear algebra: dot product, `matmul`, transpose, inverse, norms, rank and eigenvalues.
- Statistical and aggregate operations.
- Reshaping and reorganizing arrays, vertical and horizontal stacking.
- Broadcasting, boolean masking and advanced indexing, loading data from files.
- Practical examples aimed at neural network computations.

### Lab03 — The Perceptron & MLP for Logic Functions

- A `step` (threshold) activation function and a generic `neuron(X, W, b)` building block.
- Single perceptrons for the `AND`, `OR` and `AND NOT` gates.
- Multilayer perceptrons that solve `XOR` using two different decompositions.
- Implementing a Boolean function with a multilayer network.

### Lab04 — More Boolean Functions with MLPs

- Implementing the Boolean function from the lecture slides in Python.
- Building a multilayer perceptron for the four-variable `XOR` function.

### Lab05 — OOP, Lines & Perceptron Learning

- Object-Oriented Programming in Python: classes, objects, attributes and methods.
- Finding the slope and the equation of a line through two points.
- Mapping a line equation to perceptron weights and a bias (`z = w·x + b`).
- Plotting lines and multiple lines with Matplotlib.
- Visualizing the decision boundary of a neuron and the regions it separates.
- The Perceptron Learning Algorithm / Delta Rule implemented as a `Perceptron` class, trained on `AND`.

### Lab06 — Backpropagation

- The general steps of MLP learning: network initialization, forward pass, loss, backward pass, weight update.
- Sigmoid activation and its derivative.
- Training a 2-layer network on the `XOR` problem from scratch with NumPy.
- Weight and bias matrices, learning rate, epochs and the training loop.

### Lab07 — Deep Learning with Keras

- Installing Keras and building a single linear unit.
- Sequential models, `Dense` layers and hidden layers.
- Activation functions (with figures for the functions and their derivatives).
- Loss functions, optimizers, `compile()`, `fit()` and evaluating the training curves.
- Learning rate and batch size, and their effect on training.
- Normalization and why features should be scaled.
- The validation set and its role in monitoring generalization.
- Handling missing values in Pandas: `dropna`, filling numeric columns with the mean, categorical
  columns with the mode, and automating it for all columns.
- The 6 basic steps to build a neural network in Keras, applied end-to-end.

### Lab08 — Practical Models & Intro to CNNs

- Regression with Keras on the House Prices data: scaling, `Dense` layers, `mse` / `mae`, training and prediction.
- Multi-class classification on the Iris data: label encoding, one-hot encoding, `StandardScaler`,
  train/test split, softmax output and `categorical_crossentropy`.
- What a CNN is and how it differs from a fully connected network.
- CNN building blocks: convolution, activation, padding, stride, pooling, batch normalization,
  dropout, flatten, dense layers, softmax output and `model.summary()`.
- A full worked CNN example in Keras, plus useful external links and a CNN explainer.

### Lab09 — CNN Shapes, Parameters & Pooling

- Notation for input size, channels, kernel size, stride and padding.
- General formulas for the output spatial size of `Conv2D` and pooling layers.
- `valid` vs. `same` padding.
- Counting the number of parameters in convolution, flatten and dense layers.
- Step-by-step numeric examples: convolving a 5×5 image with a 3×3 filter, max pooling and average
  pooling worked out by hand.

---

## Lectures

Course slides (PDF) covering the theory behind the labs:

- Lect1 — Introduction
- Lect2 — Artificial Neural Networks: The Basics
- Lect3 — MultiLayer Perceptrons
- Lect4 — Learning the Network
- Lect5 — Learning the Network Part 2
- Lect6, Lect7, Lect8 — Backpropagation Parts 1–3

---

## Project

- `Roadmap_CNN_Project.md` — a universal, reusable roadmap for any CNN image classification project:
  problem definition, dataset acquisition and organization, preprocessing and augmentation, model
  design, training, evaluation and improvement.
- `How_to_Submit.md` — the submission mechanism and the full list of deliverables (trained model,
  source code, evaluation output, report and data documentation).

---

## Data

The `Data/` folder holds the shared datasets used by the lab notebooks (referenced as `../Data/<file>`):

- **`iris.csv`** — the classic Iris flowers dataset: sepal/petal length and width in cm with the
  `Species` label (3 classes). Used for multi-class classification in Lab08.
- **`CaliforniaHousingPrices.csv`** — California housing blocks with longitude, latitude, median house
  age, total rooms/bedrooms, population, households, median income, median house value and ocean
  proximity. Used for regression and for practicing missing-value handling.
- **`About Housing.md`** — a bilingual (English/Arabic) description of every column in the California
  housing dataset.
- **`winequality-red.csv`** — physicochemical measurements of red wine (acidity, sugar, chlorides,
  sulfur dioxide, density, pH, sulphates, alcohol) with a `quality` score. Used in the Lab07
  introduction to deep learning.
- **`telecom_churn_exam.csv`** — telecom customers with age, yearly income, plan type and service
  rating, labelled with a binary `Churn` target. Used for exam / practice classification.
- **`vehicle_maintenance_exam.csv`** — vehicle records with region code, odometer reading, purchase
  date, vehicle age and daily commute, labelled with the required `Service_Type`. Used for exam /
  practice work.

---

## Course Staff

**Course Instructor:** Dr. Hend Dawood

**Course TAs:**

- Mahmoud Esmat
- Omar Mourad
