# AI Framework: From Scratch to Llama Compatibility

A high-performance implementation of neural network architectures written in pure C. Originally designed as a pedagogical deep-dive into the mathematical foundations of AI, this framework has evolved into a functional inference engine compatible with modern LLM architectures, specifically targeting compatibility with **TinyLlama**.

## 🚀 Project Goal
The framework traces the evolution of AI from basic linear models to the state-of-the-art Transformer architecture. Its current primary objective is to provide a lightweight, dependency-free C implementation capable of running Llama-based models using efficient memory management and modern architectural primitives.

## 🛠️ Technical Stack
- **Language**: C (C11 standard)
- **Build System**: GNU Make
- **Precision Support**: 
  - `FLOAT32`: Standard precision for training and high-accuracy inference.
  - `FLOAT16`: Half-precision support for reduced memory footprint and faster inference.
- **Dependencies**: `libc` (math library `-lm`)

## 🧩 Implemented Features

### 1. High-Performance Core Engine
- **Matrix Engine**: A custom linear algebra library supporting matrix multiplication, transposition, and element-wise operations.
- **Memory Mapping (mmap)**: Implemented `model_mmap` for efficient, zero-copy loading of large model weights directly from disk.
- **Optimization**: Full implementation of SGD, Momentum, and **Adam** optimizers for training tasks.

### 2. Modern Transformer Architecture (Llama-Compatible)
The framework has been upgraded from a basic Transformer to a Llama-style architecture:
- **RMSNorm**: Replaced standard LayerNorm with Root Mean Square Layer Normalization for improved training stability and inference speed.
- **SwiGLU Activation**: Implemented the Gated Linear Unit with Swish activation in the Feed-Forward Network (FFN), replacing standard ReLU.
- **RoPE (Rotary Positional Embeddings)**: Replaced sinusoidal encoding with Rotary embeddings to better capture relative positional information.
- **Multi-Head Attention (MHA)**: Optimized QKV (Query, Key, Value) mechanism for sequence processing.
- **Causal Masking**: Integrated masking to ensure auto-regressive generation.

### 3. Inference & Generation
- **Sampling Logic**: Implementation of temperature-based sampling for diverse text generation.
- **Generation Loop**: A complete loop for iterative token prediction.
- **KV Cache**: Optimized inference by caching Key and Value tensors to avoid redundant computations.

### 4. Integration & Tooling
- **C-Bridge**: A shared library interface allowing the C engine to be invoked by higher-level languages.
- **AI CLI**: A dedicated command-line interface for interacting with the loaded models.
- **Python Bridge**: Integration for easy weight manipulation and testing via Python.

## 📖 How to Run

### Build the Project
```bash
make
```

### Run Tutorials
The project includes a series of tutorials in `bin/` that demonstrate the evolution of the architecture:
- `./bin/tut_perceptron` $ightarrow$ `./bin/tut_linear_regression` $ightarrow$ `./bin/tut_mlp_deep_dive`
- `./bin/tut_tokenizer` $ightarrow$ `./bin/tut_embedding` $ightarrow$ `./bin/tut_positional`
- `./bin/tut_attention` $ightarrow$ `./bin/tut_mha` $ightarrow$ `./bin/tut_transformer`

### Run the AI CLI
To interact with the framework via the command line:
```bash
# Example usage (depends on your build configuration)
./bin/ai_cli --model path/to/weights.bin
```

## 🗺️ Roadmap
For detailed architectural milestones and the path toward full TinyLlama compatibility, see [ROADMAP.md](ROADMAP.md) and [STATUS.md](STATUS.md).
