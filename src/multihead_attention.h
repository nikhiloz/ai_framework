#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "attention.h"
#include "matrix.h"

typedef struct {
    int num_heads;
    int head_dim;
    int embed_dim;
    AttentionLayer *heads;
    Matrix W_o; // Output projection matrix
} MultiHeadAttention;

MultiHeadAttention create_mha(int num_heads, int embed_dim);
void free_mha(MultiHeadAttention *mha);

// Computes the MHA output for a given input matrix
Matrix mha_forward(MultiHeadAttention *mha, Matrix *input);

#endif
