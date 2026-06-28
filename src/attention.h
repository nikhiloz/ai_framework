#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"

typedef struct {
    int embed_dim;
    Matrix W_q; // Weight matrix for Query
    Matrix W_k; // Weight matrix for Key
    Matrix W_v; // Weight matrix for Value
} AttentionLayer;

AttentionLayer create_attention_layer(int embed_dim);
void free_attention_layer(AttentionLayer *al);

// Computes the self-attention output for a given input matrix
// input: Matrix of shape (seq_len, embed_dim)
// returns: Matrix of shape (seq_len, embed_dim)
Matrix attention_forward(AttentionLayer *al, Matrix *input);

#endif
