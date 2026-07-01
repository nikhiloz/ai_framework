#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "attention.h"
#include "matrix.h"

typedef struct {
    int num_heads;
    int head_dim;
    int embed_dim;
    AttentionLayer *heads;
    Matrix W_q; // Global Query projection
    Matrix W_k; // Global Key projection
    Matrix W_v; // Global Value projection
    Matrix W_o; // Output projection matrix
} MultiHeadAttention;

MultiHeadAttention create_mha(int num_heads, int embed_dim);
MultiHeadAttention init_mha(int num_heads, int embed_dim);
void free_mha(MultiHeadAttention *mha);

// Computes the MHA output for a given input matrix
Matrix mha_forward(MultiHeadAttention *mha, Matrix *input);

// Backprop through MHA
// input: Original input to mha_forward (seq_len, embed_dim)
// grad_output: Gradient of loss w.r.t. output (seq_len, embed_dim)
// returns: Gradient w.r.t. input (seq_len, embed_dim)
Matrix mha_backward(MultiHeadAttention *mha, Matrix *input, Matrix *grad_output);

#endif
