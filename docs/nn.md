# 🧠 Module: Neural Network Core (`nn.c/h`)

The Neural Network core is where the "intelligence" of the framework resides. It manages the structure of the model, the flow of data (Forward Pass), and the mechanism for learning from errors (Backpropagation).

## 🎯 1. High-Level Purpose
The purpose of this module is to implement a **Multi-Layer Perceptron (MLP)**. It allows the user to define a network with any number of layers, each with a specific number of neurons and an activation function. It handles the transformation of input data into a prediction and calculates how to adjust internal weights to reduce the error of that prediction.

## 📐 2. Mathematical Foundation

### The Forward Pass (Inference)
For each layer, the network computes a weighted sum of the inputs, adds a bias, and then applies a non-linear activation function:
$$z = W \cdot x + b$$
$$a = \sigma(z)$$
Where $W$ is the weight matrix, $x$ is the input vector, $b$ is the bias, and $\sigma$ is the activation function.

### The Backward Pass (Learning)
Learning happens via the **Chain Rule** of calculus. We calculate how much the total loss $L$ changes with respect to the weights:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W}$$
This allows us to determine the "gradient"—the direction in which we should move the weights to decrease the error.

### Activation Functions
Non-linearity is what allows neural networks to solve complex problems (like XOR).
- **Sigmoid**: Squashes values between 0 and 1. Ideal for binary classification.
- **ReLU (Rectified Linear Unit)**: Returns $x$ if $x > 0$, else 0. Prevents the "vanishing gradient" problem in deep networks.
- **Tanh**: Squashes values between -1 and 1.
- **Softmax**: Turns a vector of numbers into a probability distribution (sums to 1.0).

## 🏗 3. Architectural Decisions

### Layer-Based Abstraction
We define a `Layer` struct that encapsulates its own weights, biases, and temporary storage (`z` and `activations`). This makes the network modular; adding a new layer is as simple as adding an element to the `layers` array.

### Dynamic Batch Handling
The `forward()` function is designed to handle both single samples and **mini-batches**. If the input matrix has multiple rows, the network automatically resizes its internal `z` and `activations` matrices to match the batch size, enabling efficient parallel processing.

## 🔍 4. Deep Dive into Functions

### `forward(Network *net, Matrix *input)`
- **Input**: The network and an input matrix.
- **Logic**: Iterates through layers. It multiplies input by weights, adds bias, and applies the activation function. The output of layer $i$ becomes the input for layer $i+1$.
- **Output**: The final layer's `activations` matrix contains the network's prediction.

### `train_step(...)`
- **Input**: Network, input, target, learning rate, loss type, and optimizer.
- **Logic**: 
    1. Performs a `forward()` pass.
    2. Computes the "Delta" ($\delta$) for the output layer using the `loss_gradient`.
    3. Propagates the $\delta$ backward through the layers.
    4. Uses the `Optimizer` to update weights and biases based on the calculated gradients.
- **Output**: Updates the internal weights of the network.

### `activate()` and `activate_derivative()`
- These functions implement the mathematical formulas for the activation functions and their derivatives. The derivative is essential for backpropagation because it tells the network how to "tweak" $z$ to change the output $a$.

## ⏱ 5. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
| :--- | :--- | :--- |
| `forward` | $O(L \cdot N^2)$ | $O(L \cdot N)$ |
| `train_step` | $O(L \cdot N^2)$ | $O(L \cdot N)$ |
| `activate` | $O(1)$ | $O(1)$ |

*(Where $L$ is number of layers and $N$ is average neurons per layer)*

## 🧪 6. Expected Output & Verification

### Example: Forward Pass (Identity/Linear)
**Input**: `[0.5, -0.2]`
**Weights**: `[[1.0], [2.0]]`
**Bias**: `[0.1]`
**Calculation**: $(0.5 	imes 1.0) + (-0.2 	imes 2.0) + 0.1 = 0.5 - 0.4 + 0.1 = 0.2$
**Expected Result**: `[0.2000]`

### Example: Learning Behavior (Backprop)
If the target is `[1.0]` and the prediction is `[0.2]`:
1. The **Error** is $0.2 - 1.0 = -0.8$.
2. The **Gradient** will be negative.
3. The **Update** will increase the weights/bias to push the next prediction closer to $1.0$.

## 🔗 7. Inter-dependency Map
- **Used by**: `examples/tutorials/*.c` (Model definition and training).
- **Depends on**: 
    - `matrix.h`: For all tensor operations.
    - `optimizer.h`: To apply the gradients calculated during `train_step`.
