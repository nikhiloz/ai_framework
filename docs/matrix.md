# 🧱 Module: Matrix Engine (`matrix.c/h`)

The Matrix Engine is the fundamental bedrock of the AI Framework. In deep learning, almost every operation—from the forward pass to the backpropagation—is essentially a matrix operation. This module provides the linear algebra primitives required to handle these computations efficiently.

## 🎯 1. High-Level Purpose
The goal of the Matrix Engine is to provide a way to store and manipulate multi-dimensional arrays (Tensors) of floating-point numbers. Instead of using nested arrays (e.g., `float**`), which are slow and fragment memory, this engine uses a **contiguous 1D array** to simulate a 2D matrix.

## 📐 2. Mathematical Foundation

### Matrix Multiplication (The Dot Product)
The core of any neural network is the multiplication of an input vector by a weight matrix. For two matrices $A$ (size $m 	imes n$) and $B$ (size $n 	imes p$), the resulting matrix $C$ (size $m 	imes p$) is calculated as:
$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$
This operation transforms the input data into a new feature space.

### Matrix Transposition
Transposition flips a matrix over its diagonal. If $A$ is $m 	imes n$, $A^T$ is $n 	imes m$. This is critical during **Backpropagation**, where we must multiply the error delta by the transposed weight matrix to propagate the error backward.

### Broadcasting (Bias Addition)
In our framework, we implement a specific form of broadcasting. When adding a bias vector $\vec{b}$ (size $1 	imes n$) to a matrix $Z$ (size $m 	imes n$), the bias is added to **every single row** of $Z$.
$$Z_{ij} = Z_{ij} + b_j$$

## 🏗 3. Architectural Decisions

### Row-Major Memory Layout
We store matrices in a single block of memory. To access an element at row `i` and column `j`, we use the formula:
`index = i * cols + j`
**Why?**
- **Cache Locality**: Contiguous memory is significantly faster for the CPU to read.
- **Simple Allocation**: Only one `malloc` call is needed per matrix.

### Dynamic Lifecycle
Matrices are created on the heap. Every `create_matrix` call must eventually be paired with a `free_matrix` call to prevent memory leaks, which are fatal in training loops that run for thousands of epochs.

## 🔍 4. Deep Dive into Functions

### `matrix_multiply(Matrix *a, Matrix *b)`
- **Input**: Two matrices where `a->cols == b->rows`.
- **Logic**: A triple-nested loop. The outer two loops iterate over the resulting matrix dimensions, and the innermost loop calculates the dot product.
- **Output**: A new `Matrix` object.

### `matrix_add_bias(Matrix *a, Matrix *bias, Matrix *result)`
- **Input**: A matrix $A$ and a bias vector $B$.
- **Logic**: It iterates through every row of $A$ and adds the corresponding element of $B$ to each column.
- **Output**: The result is stored in the `result` matrix.

### `matrix_transpose(Matrix *m)`
- **Input**: A matrix $M$.
- **Logic**: It creates a new matrix and maps `res[j][i] = m[i][j]`.
- **Output**: A new transposed `Matrix`.

## ⏱ 5. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
| :--- | :--- | :--- |
| `create_matrix` | $O(1)$ | $O(rows 	imes cols)$ |
| `matrix_multiply` | $O(m 	imes n 	imes p)$ | $O(m 	imes p)$ |
| `matrix_transpose` | $O(rows 	imes cols)$ | $O(rows 	imes cols)$ |
| `matrix_add_bias` | $O(rows 	imes cols)$ | $O(1)$ (in-place result) |

## 🧪 6. Expected Output & Verification

If you were to implement a small test for this module, here is what you should expect:

### Example: Multiplication
**Input A**: `[[1, 2], [3, 4]]` (2x2)
**Input B**: `[[5, 6], [7, 8]]` (2x2)
**Expected Result**:
`[ (1*5 + 2*7), (1*6 + 2*8) ]` $ightarrow$ `[19, 22]`
`[ (3*5 + 4*7), (3*6 + 4*8) ]` $ightarrow$ `[43, 50]`

### Example: Bias Addition
**Input Matrix**: `[[1, 1], [1, 1]]`
**Bias Vector**: `[10, 20]`
**Expected Result**:
`[[11, 21], [11, 21]]`

## 🔗 7. Inter-dependency Map
- **Used by**: `nn.c` (Forward pass and Backprop), `optimizer.c` (Weight updates).
- **Depends on**: `stdlib.h` (for `malloc`/`free`).
