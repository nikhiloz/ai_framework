#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include "matrix.h"
#include "model.h"

typedef enum {
    GGUF_TENSOR_F32 = 0,
    GGUF_TENSOR_F16 = 1,
    GGUF_TENSOR_Q4_0 = 2,
    GGUF_TENSOR_Q4_1 = 3,
    GGUF_TENSOR_Q5_0 = 6,
    GGUF_TENSOR_Q5_1 = 7,
    GGUF_TENSOR_Q8_0 = 8,
    GGUF_TENSOR_Q8_1 = 9,
    GGUF_TENSOR_Q2_K = 10,
    GGUF_TENSOR_Q3_K = 11,
    GGUF_TENSOR_Q4_K = 12,
    GGUF_TENSOR_Q5_K = 13,
    GGUF_TENSOR_Q6_K = 14,
    GGUF_TENSOR_Q8_K = 15,
    GGUF_TENSOR_UNKNOWN = -1
} GGUF_TensorType;

typedef struct {
    char name[256];
    uint64_t offset;
    uint64_t elements;
    GGUF_TensorType type;
} GGUF_TensorInfo;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_pair_count;
    GGUF_TensorInfo *tensors;
    uint64_t tensor_data_offset;
    uint32_t alignment;
    
    int embed_dim;
    int num_blocks;
    int num_heads;
    int head_dim;
    int ffn_dim;
    int vocab_size;
} GGUF_ModelInfo;

GGUF_ModelInfo* gguf_parse_header(const char* filename);
void gguf_free_model_info(GGUF_ModelInfo* info);

// New Phase 2 Functions
TransformerModel load_model_gguf(const char* filename);

#endif
