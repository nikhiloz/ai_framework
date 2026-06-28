# Roadmap to Large Language Models (LLMs)

This document outlines the gap between the current Multi-Layer Perceptron (MLP) framework and a conversational model capable of being run by Ollama.

## 1. Current State: The MLP
Our current framework implements a basic Feed-Forward Neural Network. It is the foundation of AI, handling basic vector-to-vector mappings (like the XOR problem).

## 2. The Gap: MLP vs. Transformer
To reach "conversation level," we must move from simple dense layers to the **Transformer Architecture**.

| Feature | MLP (Current) | LLM / Transformer |
| :--- | :--- | :--- |
| **Architecture** | Dense Layers | Transformer Blocks |
| **Memory** | No temporal memory | Self-Attention Mechanism |
| **Input** | Simple Vectors | Tokenized Text |
| **Scale** | 10s of Parameters | Billions of Parameters |
| **Format** | Custom C Structs | GGUF (Ollama Format) |

## 3. The Path Forward

### Stage 1: Language Foundation
- [x] **Tokenization**: Implement a system to convert text to token IDs (e.g., BPE).
- [x] **Embedding Layers**: Map token IDs to high-dimensional vectors.

### Stage 2: Transformer Architecture
- [x] **Self-Attention**: Implement the mechanism that allows the model to weight the importance of different tokens in a sequence.
- [x] **Multi-Head Attention**: Parallel attention heads for diverse feature extraction.
- [x] **Positional Encoding**: Adding information about the order of tokens.
- [x] **Layer Normalization**: Ensuring numerical stability during training.
- [ ] **Transformer Block**: Integration of the above into a repeatable layer. (In progress/Integrated)
- [ ] **Causal Masking**: Training for "Next Token Prediction."

### Stage 3: Training & Scaling
- **Causal Masking**: Training for "Next Token Prediction."
- **Performance**: Integrating BLAS or CUDA for GPU acceleration.

### Stage 4: Ollama Integration
- **GGUF Exporter**: Developing a tool to export weights into the GGUF binary format used by `llama.cpp` and Ollama.

## 4. Conclusion
The current framework has mastered the "alphabet" of AI (Weights $ightarrow$ Forward $ightarrow$ Loss $ightarrow$ Backprop $ightarrow$ Update). The next leap is moving from static mappings to sequence-aware attention.
