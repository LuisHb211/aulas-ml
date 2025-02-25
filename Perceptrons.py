import numpy as np

digits = {
    0: [-1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1],
    1: [-1, -1,  1, -1, -1,
        -1,  1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1, -1,  1, -1, -1,
        -1,  1,  1,  1, -1],
    2: [-1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1, -1, -1,  1, -1,
        -1,  1, -1, -1, -1,
         1,  1,  1,  1,  1],
    3: [-1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1, -1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1],
    4: [ 1, -1, -1,  1, -1,
         1, -1, -1,  1, -1,
         1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1,
        -1, -1, -1,  1, -1],
    5: [ 1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1,
        -1, -1, -1, -1,  1,
         1,  1,  1,  1, -1],
    6: [-1,  1,  1,  1, -1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1],
    7: [ 1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1,
        -1, -1,  1, -1, -1,
        -1,  1, -1, -1, -1,
         1, -1, -1, -1, -1],
    8: [-1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1],
    9: [-1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1]
}


X = np.array(list(digits.values()))
T = np.eye(10) * 2 - 1  # Saídas esperadas em bipolar (-1 e 1)

n_inputs = 25
n_outputs = 10
W = np.random.randn(n_outputs, n_inputs) * 0.01
B = np.random.randn(n_outputs) * 0.01

def activation(x):
    return np.where(x >= 0, 1, -1)

# Treinamento Perceptron
learning_rate = 1.0
max_epochs = 100
for epoch in range(max_epochs):
    error_count = 0
    for i in range(len(X)):
        x_i = X[i]
        target = T[i]
        y_in = np.dot(W, x_i) + B
        y = activation(y_in)
        
        if not np.array_equal(y, target):
            W += learning_rate * np.outer(target - y, x_i)
            B += learning_rate * (target - y)
            error_count += 1
    
    if error_count == 0:
        break

def classify_digit(digit_array):
    y_in = np.dot(W, digit_array) + B
    y = activation(y_in)
    return np.argmax(y), W, B

test_digit = digits[3]  
recognized_digit, weights, biases = classify_digit(test_digit)
print("Reconhecido como:", recognized_digit)
print("Pesos dos neurônios:\n", weights)
print("Bias dos neurônios:\n", biases)
