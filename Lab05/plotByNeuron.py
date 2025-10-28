import numpy as np
import matplotlib.pyplot as plt


# === Define neuron and perceptron logic ===
def step(x):
    return 1 if x >= 0 else 0


def neuron(x, w, b):
    return step(np.dot(x, w) + b)


def perceptron(x):
    w = np.array([1, 1])
    b = -1
    n1 = neuron(x, w, b)
    return n1


points = np.random.randint(-6, 6, size=(50, 2))

# Compute perceptron outputs
outputs = np.array([[perceptron(p) for p in points]])
# print(outputs.shape)
# === Create grid for visualization ===
limitPoint = 6
# Create a grid of points
x = np.linspace(-limitPoint, limitPoint, 200)
y = np.linspace(-limitPoint, limitPoint, 200)
X, Y = np.meshgrid(x, y)
Xf = X.flatten()
Yf = Y.flatten()
# flatten() converts a multidimensional array into a 1D array.
Z = np.array([[perceptron(p) for p in zip(Xf, Yf)]])
Z = Z.reshape(X.shape)

# Plot decision boundary
plt.contour(X, Y, Z, levels=[0], colors=['blue'], linewidths=2)

# Highlight region where neuron output == 1
plt.contourf(X, Y, Z, levels=[-0.5, 0.5], colors=['yellow'], alpha=0.5)

# Optionally scatter sample points
# plt.scatter(X[Z == 1], Y[Z == 1], s=90, color='green')
plt.scatter(points[:, 0], points[:, 1], c=outputs, cmap='bwr_r', s=200, edgecolors='k')

plt.title('Neuron decision region: output = 1 highlighted')
# plt.grid(True)
margin = 1
plt.xlim(-limitPoint - margin, limitPoint + margin)
plt.ylim(-limitPoint - margin, limitPoint + margin)
plt.xlabel('x', loc='left')
plt.ylabel('y', loc='bottom')
# plt.axis('equal')  # Equal aspect ratio
# plt.legend()
# plt.margins(x=-50, y=15)

# Get current axes
ax = plt.gca()
# Move left and bottom spines (axes lines) to the center
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

plt.show()
