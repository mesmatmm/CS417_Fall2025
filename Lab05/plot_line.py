import matplotlib.pyplot as plt
import numpy as np

# Define x range
x = np.linspace(0, 1, 5)
# linspace is a NumPy function that generates a sequence of evenly
# spaced numbers between two endpoints.
print(x)
# Calculate y from the line equation x + y = 1
y = 1 - x

# Plot the line
plt.plot(x, y, label='x + y = 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Straight Line')
plt.legend()
plt.grid(True)
# plt.axis('equal')  # Equal aspect ratio
plt.show()
