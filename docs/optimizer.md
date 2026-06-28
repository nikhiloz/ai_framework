# 🧭 Module: Optimizer System (`optimizer.c/h`)

The Optimizer is the "navigator" of the neural network. While the Neural Network core calculates the gradients (which direction is "downhill" on the error surface), the Optimizer decides exactly how to take that step to reach the minimum loss as quickly and stably as possible.

## 🎯 1. High-Level Purpose
The goal of this module is to decouple the **calculation of gradients** from the **updating of weights**. By separating these, we can switch between different optimization algorithms (like SGD or Adam) without changing a single line of code in the neural network's backpropagation logic.

## 📐 2. Mathematical Foundation

### Stochastic Gradient Descent (SGD)
The simplest form of optimization. It updates weights by taking a small step in the opposite direction of the gradient:
$$W_{t+1} = W_t - \eta \cdot 
abla L(W_t)$$
Where $\eta$ is the learning rate and $
abla L$ is the gradient.

### Momentum
SGD often oscillates wildly in "ravines" of the loss surface. Momentum solves this by adding a fraction of the previous update to the current one, creating a "velocity" effect:
$$v_{t+1} = \beta \cdot v_t + (1 - \beta) \cdot 
abla L(W_t)$$
$$W_{t+1} = W_t - \eta \cdot v_{t+1}$$
This helps the optimizer push through flat regions and dampen oscillations.

### ADAM (Adaptive Moment Estimation)
Adam is the industry standard. It calculates an individual learning rate for every single parameter by tracking both the **mean** (first moment) and the **uncentered variance** (second moment) of the gradients:
1. **First Moment**: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. **Second Moment**: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. **Bias Correction**: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
4. **Update**: $W_{t+1} = W_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## 🏗 3. Architectural Decisions

### State Persistence
Unlike SGD, which is "stateless," Momentum and Adam need to remember the previous gradients. To handle this, the `Optimizer` struct contains arrays of matrices (`weight_m`, `weight_v`, etc.). 
- Each layer in the network has a corresponding "moment" matrix in the optimizer.
- This ensures that the "velocity" of weight #42 in layer 2 is tracked independently from weight #1 in layer 1.

### Decoupled Interface
The `optimizer_step` function takes the gradient as an input. This means the `nn.c` file only cares about *what the gradient is*, while the `optimizer.c` file cares about *how to use it*.

## 🔍 4. Deep Dive into Functions

### `create_optimizer(Network *net, OptimizerType type, float lr)`
- **Logic**: Allocates the `Optimizer` struct and, depending on the type, allocates the necessary state matrices.
- **Memory**: If `ADAM` is selected, it allocates four matrices per layer (two for weights, two for biases). If `SGD` is selected, no extra matrices are allocated.

### `optimizer_step(Optimizer *opt, Network *net, Matrix *w_grad, Matrix *b_grad, int layer_idx)`
- **Logic**:
    - For **SGD**: Directly subtracts `lr * grad` from weights.
    - For **Momentum**: Updates the velocity matrix $m$ and then updates weights.
    - For **Adam**: Updates both $m$ and $v$, applies bias correction, and performs the adaptive update.
- **Output**: Modifies the weights and biases of the specified layer in-place.

### `free_optimizer(Optimizer *opt)`
- **Logic**: Iterates through all layers and frees the state matrices and the pointer arrays. This is critical to avoid massive memory leaks during hyperparameter tuning.

## ⏱ 5. Complexity Analysis

| Algorithm | Time Complexity (per step) | Space Complexity (per layer) |
| :--- | :--- | :--- |
| **SGD** | $O(W + B)$ | $O(1)$ |
| **Momentum** | $O(W + B)$ | $O(W + B)$ |
| **Adam** | $O(W + B)$ | $O(2(W + B))$ |

*(Where $W$ is number of weights and $B$ is number of biases)*

## 🧪 6. Expected Output & Verification

### Convergence Comparison
If you train the XOR model with different optimizers, you will notice:
- **SGD**: May take 10,000+ epochs or get stuck in a local minimum.
- **Momentum**: Converges faster than SGD and is less likely to get stuck.
- **Adam**: Often converges in under 1,000 epochs with very high precision.

### Weight Update Analysis
In a simple SGD step:
**Weight**: `0.5`, **Gradient**: `0.1`, **LR**: `0.01`
**Calculation**: $0.5 - (0.01 	imes 0.1) = 0.499$
**Result**: The weight decreased slightly.

## 🔗 7. Inter-dependency Map
- **Used by**: `nn.c` (inside `train_step`).
- **Depends on**:
    - `matrix.h`: For scalar multiplication and matrix access.
    - `nn.h`: To access the layer weights and biases that need updating.
