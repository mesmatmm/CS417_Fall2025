import numpy as np


################################################################################
# Step activation
def step(x):
    return 1 if x >= 0 else 0


################################################################################
# Generic neuron
def neuron(X, W, b):
    return step(np.dot(X, W) + b)


################################################################################
def XOR_MLP1(x):  # A XOR B = (X OR Y) AND ((NOT X) OR (NOT Y))
    # ---- Hidden layer ----
    # N1 : computes X OR Y
    h1 = neuron(x, np.array([1, 1]), -1)
    # N2 : computes (NOT X) OR (NOT Y)
    h2 = neuron(x, np.array([-1, -1]), 1)

    # ---- Output layer ----
    # Combine them: XOR = (X OR Y) AND (NOT X OR NOT Y)
    out = neuron(np.array([h1, h2]), np.array([1, 1]), -2)
    return out


################################################################################
def xor_mlp2(x):  # A XOR B = (X AND (NOT Y)) OR ((NOT X) OR Y)
    # ---- Hidden layer ----
    n1 = neuron(x, np.array([-1, 1]), -1)
    n2 = neuron(x, np.array([1, -1]), -1)

    # Output Layer
    n3 = neuron(np.array([n1, n2]), np.array([1, 1]), -1)

    return n3


################################################################################
def and_perceptron(x):
    return neuron(x, np.array([1, 1]), -2)


def or_perceptron(x):
    return neuron(x, np.array([2, 2]), -2)


def and_not_perceptron(x):
    return neuron(x, np.array([2, -1]), -2)


################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
def main():
    ###############################################################
    # Test all 4 input pairs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # for x in X:
    #     print(f"{x} -> AND = {and_perceptron(x)}")
    # print("----------------------------------------")
    # for x in X:
    #     print(f"{x} -> OR = {or_perceptron(x)}")
    # print("----------------------------------------")
    # for x in X:
    #     print(f"{x} -> AND_NOT = {and_not_perceptron(x)}")
    # print("----------------------------------------")
    for x in X:
        print(f"{x} -> XOR_MLP1 = {XOR_MLP1(x)}, XOR_MLP2 = {xor_mlp2(x)}, XOR = {x[0] ^ x[1]}")
    ####################################################################################################
    X = [[a, b, c, d] for a in range(2) for b in range(2)
         for c in range(2) for d in range(2)]
    X = np.array(X)
    for x in X:
        print(f"#Test HERE")
    ####################################################################################################


# if __name__ == __main__: (this script can be imported OR run standalone)
# Functions and classes in this module can be reused without the main block of code executing
if __name__ == '__main__':
    main()
