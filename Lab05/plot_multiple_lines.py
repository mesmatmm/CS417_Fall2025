import matplotlib.pyplot as plt
import numpy as np

# Define x range
x = np.linspace(-1, 1, 100)

# Calculate y for the first line: x + y = 1
y1 = 1 - x

# Calculate y for the second line: x - y = 0
y2 = x

# Plot both lines
plt.plot(x, y1, label='x + y = 1', color='blue')
plt.plot(x, y2, label='x - y = 0', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multiple Lines')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.show()