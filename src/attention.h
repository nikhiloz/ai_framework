#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"
#include <stdbool.h>

typedef struct {
    int embed_dim;
    
    // KV Cache
    Matrix *K_cache; // Cached keys (accumulated over time)
    Matrix *V_cache; // Cached values (accumulated over time)
    int cache_len;
} AttentionLayer;

AttentionLayer create_attention_layer(int embed_dim);
AttentionLayer init_attention_layer(int embed_dim);
void free_attention_layer(AttentionLayer *al);

// Computes the self-attention output for a given Q, K, V
// Q, K, V: Matrices of shape (seq_len, embed_dim)
// mask: If true, applies a causal look-ahead mask (for generative models)
// returns: Matrix of shape (seq_len, embed_dim)
Matrix attention_forward(AttentionLayer *al, Matrix *Q, Matrix *K, Matrix *V, bool mask);

// Backprop through attention
// input: The original input to attention_forward (seq_len, embed_dim)
// grad_output: Gradient of loss w.r.t. output (seq_len, embed_dim)
// mask: Must match the mask used in forward pass
// returns: Gradient w.r.t. input (seq_len, embed_dim)
Matrix attention_backward(AttentionLayer *al, Matrix *input, Matrix *grad_output, bool mask);

#endif
