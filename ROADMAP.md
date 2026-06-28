# Roadmap to Conversational AI

This document outlines the phased evolution of the `ai_framework` from a training tutorial to an operational conversational agent.

## Phase 1: Foundation & Training
- [x] Matrix Engine & MLP Foundation
- [x] Transformer Architecture & Causal Masking
- [x] End-to-End Training Harness (Synthetic)

## Phase 2: From Training to Generation (Inference)
- [x] **Tokenizer Integration:** Implement a robust BPE/SentencePiece tokenizer to handle real text.
- [x] **Generation Loop (Sampling):** Implement `greedy` and `top-k` sampling to generate text token-by-token.
- [x] **KV Cache:** Implement Key-Value caching to make inference efficient.

## Phase 3: Conversational Capabilities
- [x] **Instruction Fine-Tuning Pipeline:** Train on instruction-response pairs.
- [x] **Prompt Formatting:** Handle chat history and chat templates.

## Phase 4: Operational Integration
- [x] **Model Weight Export:** Implement GGUF exporter for compatibility with `llama.cpp` / Ollama.
- [x] **Conversational Web Interface:** Integrate the operational model into the webserver.
- [x] **C-Bridge & Shared Library:** Create `.so` for Python interop.
- [x] **Interactive AI CLI:** Implement a streaming CLI for local interaction.

## Phase 5: Training & Personality (Current Goal)
- [ ] **Dataset Curation:** Gather high-quality "Jolly" conversational pairs.
- [ ] **Pre-training Loop:** Implement large-scale text prediction to learn language.
- [ ] **SFT (Supervised Fine-Tuning):** Train the model to follow instructions.
- [ ] **Personality Alignment:** Fine-tune the model specifically on happy, high-energy datasets to create the "Jolly" persona.
