# AI Framework: Llama.cpp Engine

This framework was originally designed as a pedagogical deep-dive into the mathematical foundations of AI, implemented in pure C. It has now been updated to use the high-performance **llama.cpp** inference engine for running modern LLM architectures like TinyLlama, significantly improving performance, stability, and compatibility.

## 🚀 Project Goal
To provide a lightweight, high-performance C framework for running Llama-based models using efficient memory management and modern architectural primitives, powered by the `llama.cpp` inference engine.

## 🛠️ Technical Stack
- **Inference Engine**: `llama.cpp` (production-ready engine)
- **Framework Core**: C (C11 standard)
- **Integration**: Python bridge for CLI interaction
- **Build System**: GNU Make & CMake (for `llama.cpp`)

## 🧩 Key Features
- **llama.cpp Backend**: Replaced custom, unstable C-inference engine with the highly optimized `llama.cpp` library.
- **GGUF Support**: Native support for loading GGUF format models.
- **Incremental Generation**: Optimized token generation pipeline for efficient inference.
- **Python-C Bridge**: Seamless bridge for CLI interactions and model management.

## 📖 How to Run

### Build the Project
```bash
make clean && make
```

### Run the AI CLI
To interact with the framework via the command line:
```bash
./run_ai.sh
```
*(Ensure `llama.cpp` dependencies are built as per the documentation in `llama.cpp/`)*

## 🗺️ Roadmap
For detailed architectural milestones, see [ROADMAP.md](ROADMAP.md) and [STATUS.md](STATUS.md).
