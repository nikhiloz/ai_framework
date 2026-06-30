# Neural Networks for C Programmers: An Implementation Guide

Welcome to the `ai_framework`. If you are proficient in C but new to neural networks, this guide is designed for you. Instead of abstract mathematics, we focus on how AI concepts are concretely implemented using structs, pointers, and memory blocks.

## 🗺️ Concept $ightarrow$ Code Mapping

To understand the framework, start with this mapping of AI concepts to their specific implementations in the `src/` directory.

| AI Concept | C Implementation File | What to look for in the code |
| :--- | :--- | :--- |
| **Tensors / Matrices** | `matrix.c / .h` | The `Matrix` struct and `void* data` for polymorphism. |
| **Layer Primitives** | `nn.c / .h`, `rmsnorm.c / .h` | Linear transformations and normalization functions. |
| **Word Embeddings** | `embedding.c / .h` | Lookups mapping `int` token IDs to `Matrix` vectors. |
| **Self-Attention** | `attention.c / .h` | The QKV (Query, Key, Value) dot-product logic. |
| **MHA (Multi-Head)** | `multihead_attention.c / .h` | Splitting and merging attention heads. |
| **Transformer Block** | `transformer_block.c / .h` | The composition of MHA $ightarrow$ RMSNorm $ightarrow$ SwiGLU. |
| **Model Architecture** | `model.c / .h` | The `TransformerModel` struct aggregating blocks. |
| **Weight Loading** | `model_mmap.c` | The use of `mmap()` for zero-copy weight access. |
| **Tokenization** | `tokenizer.c / .h` | Mapping raw strings to integer arrays. |
| **Inference/Generation** | `prediction.c / .h`, `sampling.c / .h` | The loop that predicts one token at a time. |

---

## 🧠 1. The "Tensor" in C

In AI, everything is a tensor (a multi-dimensional array). In this framework, we simplify this to a **Matrix**.

### Technical Implementation: `matrix.h`
We use a flexible `Matrix` struct to handle different data precisions (`FLOAT32` vs `FLOAT16`) without duplicating code.

```c
typedef struct {
    int rows;
    int cols;
    Precision precision;
    void *data; // Points to either float* or uint16_t*
} Matrix;
```

**Key Insight for C Programmers:**
The `void *data` pointer allows the core matrix multiplication functions to handle different bit-widths. You will see `get_val()` and `set_val()` helpers used throughout the codebase to abstract the precision casting.

---

## 🏗️ 2. The Architecture Hierarchy

The system is built as a nested hierarchy. A model is a collection of blocks, and a block is a collection of layers and weights.

![Memory Hierarchy](docs/diagrams/arch_memory.png)

### Hierarchy Breakdown:
1.  **`TransformerModel`**: The top-level container. It manages the tokenizer, embeddings, and an array of `TransformerBlock` structs.
2.  **`TransformerBlock`**: The fundamental unit of the Transformer. It coordinates the flow between Attention and the Feed-Forward Network.
3.  **`MultiHeadAttention`**: Implements the logic that allows the model to "attend" to different parts of a sequence simultaneously.
4.  **`Matrix`**: The leaf node. Everything eventually boils down to a `Matrix` multiplication.

---

## 🔄 3. The Forward Pass (How data flows)

When you pass a token into the model, it undergoes a series of transformations. The `transformer_block_forward` function is the heart of this process.

![Data Flow](docs/diagrams/arch_flow.png)

### The Execution Pipeline:
1.  **RMSNorm (ln1):** The input is normalized. This prevents gradients from exploding and keeps values in a range that the activation functions can handle.
2.  **Multi-Head Attention (MHA):** The model computes the relationship between the current token and all previous tokens.
3.  **Residual Connection 1:** The output of MHA is added back to the original input (`x + MHA(x)`). This allows gradients to flow more easily during training.
4.  **RMSNorm (ln2):** Another normalization step.
5.  **SwiGLU FFN:** A gated linear unit. It's essentially two matrix multiplications where one "gates" the other using a sigmoid-like function.
6.  **Residual Connection 2:** The final FFN output is added to the previous residual stream.

---

## 💾 4. Memory Management & Weights

Unlike simple C programs that use `fread()` to load data, this framework uses **Memory Mapping (`mmap`)**.

### Why `mmap`? (`model_mmap.c`)
Large models (like TinyLlama) have millions of parameters. Loading them all into RAM via `malloc` and `fread` is slow and consumes massive memory.
`mmap()` tells the OS to map the binary file on disk directly into the process's address space. The OS loads the weights "on demand" (page-faulting), which:
1.  **Reduces Startup Time:** The model "loads" instantly.
2.  **Optimizes RAM:** The OS can share the memory across processes and drop pages that aren't being used.

---

## 🛠️ 5. How to start exploring the code

If you want to understand how a specific part works, follow this suggested path:

1.  **Start with `matrix.c`**: Understand how `matrix_multiply` is implemented.
2.  **Move to `rmsnorm.c`**: See how a simple vector normalization is done.
3.  **Study `transformer_block.c`**: Trace the `forward` function and see how it calls the other modules.
4.  **End with `model.c`**: See how the blocks are sequenced to create a full model.

---

## 🚀 Summary for the C Developer
Neural networks in C are essentially **highly structured memory management** combined with **loops over floating-point arrays**. If you can manage a pointer to a block of memory and implement a nested loop for matrix multiplication, you have already mastered the hardest part of the implementation.
