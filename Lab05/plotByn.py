import numpy as np
import matplotlib.pyplot as plt

# Define the neuron parameters
w1, w2, b = 1, 1, -1  # corresponds to the line: x + y - 1 = 0

limitPoint = 6
# Create a grid of points
x = np.linspace(-limitPoint, limitPoint, 100)
y = np.linspace(-limitPoint, limitPoint, 100)
X, Y = np.meshgrid(x, y)

# print(X)
# print(Y)
# print(X.shape)
# print(Y.shape)

# Compute neuron output
Z = w1 * X + w2 * Y + b
output = (Z <= 0).astype(int)  # Step function applied elementwise # step function 1 for True and 0 for False

# Plot decision boundary
plt.contour(X, Y, Z, levels=[0], colors=['blue'], linewidths=2)

# Highlight region where neuron output == 1
plt.contourf(X, Y, output, levels=[0.5, 1.5], colors=['yellow'], alpha=0.5)

# Optionally scatter sample points
# plt.scatter(X[output == 1], Y[output == 1], s=5, color='green')

plt.title('Neuron decision region: output = 1 highlighted')
plt.grid(True)
plt.xlim(-limitPoint, limitPoint)
plt.ylim(-limitPoint, limitPoint)
plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')  # Equal aspect ratio
# plt.legend()

# Get current axes
ax = plt.gca()
# Move left and bottom spines (axes lines) to the center
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

plt.show()
