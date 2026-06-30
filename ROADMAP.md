# Roadmap to Conversational AI

This document outlines the evolution of the `ai_framework` from a custom training framework to a functional inference engine compatible with Llama-based models like TinyLlama.

## Phase 1: Foundation & Training
- [x] Matrix Engine & MLP Foundation
- [x] Transformer Architecture & Causal Masking
- [x] End-to-End Training Harness (Synthetic)

## Phase 2: From Training to Generation (Inference)
- [x] Tokenizer Integration
- [x] Generation Loop (Sampling)
- [x] KV Cache

## Phase 3: Conversational Capabilities
- [x] Instruction Fine-Tuning Pipeline
- [x] Prompt Formatting

## Phase 4: Operational Integration
- [x] Model Weight Export
- [x] Conversational Web Interface
- [x] C-Bridge & Shared Library
- [x] Interactive AI CLI

## Phase 5: TinyLlama Compatibility (Current Goal)
- [x] **RMSNorm Implementation:** Replace standard LayerNorm.
- [x] **SwiGLU Activation:** Replace ReLU in FFN.
- [x] **RoPE Implementation:** Implement Rotary Positional Embeddings.
- [x] **Weight Loader:** Develop a parser for Llama-compatible weight files.
- [x] **Inference Verification:** Run TinyLlama inference on sample prompts.
