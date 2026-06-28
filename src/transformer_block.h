#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "multihead_attention.h"
#include "layernorm.h"
#include "matrix.h"

typedef struct {
    MultiHeadAttention mha;
    LayerNorm ln1;
    LayerNorm ln2;
    
    // Feed-Forward Network (FFN)
    Matrix W1; // (embed_dim, ffn_dim)
    Matrix b1; // (1, ffn_dim)
    Matrix W2; // (ffn_dim, embed_dim)
    Matrix b2; // (1, embed_dim)
    int embed_dim;
    int ffn_dim;

    // Intermediate activations for backward pass
    Matrix ln1_input;
    Matrix mha_out;
    Matrix res1;
    Matrix ln2_input;
    Matrix ffn1;
} TransformerBlock;

TransformerBlock create_transformer_block(int num_heads, int embed_dim, int ffn_dim);
void free_transformer_block(TransformerBlock *tb);

// Forward pass through the transformer block
// input: Matrix of shape (seq_len, embed_dim)
// returns: Matrix of shape (seq_len, embed_dim)
Matrix transformer_block_forward(TransformerBlock *tb, Matrix *input);

// Backprop through the transformer block
// input: Original input to forward (seq_len, embed_dim)
// grad_output: Gradient w.r.t. block output (seq_len, embed_dim)
// returns: Gradient w.r.t. block input (seq_len, embed_dim)
Matrix transformer_block_backward(TransformerBlock *tb, Matrix *input, Matrix *grad_output);

#endif
