#include "transformer_block.h"
#include <stdlib.h>
#include <math.h>

TransformerBlock create_transformer_block(int num_heads, int embed_dim, int ffn_dim) {
    TransformerBlock tb;
    tb.embed_dim = embed_dim;
    tb.ffn_dim = ffn_dim;
    
    tb.mha = create_mha(num_heads, embed_dim);
    tb.ln1 = create_layernorm(embed_dim);
    tb.ln2 = create_layernorm(embed_dim);
    
    tb.W1 = create_matrix(embed_dim, ffn_dim);
    tb.b1 = create_matrix(1, ffn_dim);
    tb.W2 = create_matrix(ffn_dim, embed_dim);
    tb.b2 = create_matrix(1, embed_dim);
    
    matrix_fill_random(&tb.W1);
    matrix_fill_random(&tb.W2);
    matrix_fill_zero(&tb.b1);
    matrix_fill_zero(&tb.b2);
    
    return tb;
}

void free_transformer_block(TransformerBlock *tb) {
    free_mha(&tb->mha);
    free_layernorm(&tb->ln1);
    free_layernorm(&tb->ln2);
    free_matrix(&tb->W1);
    free_matrix(&tb->b1);
    free_matrix(&tb->W2);
    free_matrix(&tb->b2);
}

Matrix transformer_block_forward(TransformerBlock *tb, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;
    
    // 1. Multi-Head Attention + Residual Connection + LayerNorm
    Matrix ln1_input = create_matrix(seq_len, embed_dim);
    // Copy input for LayerNorm (since it's in-place)
    for(int i=0; i < seq_len * embed_dim; i++) ln1_input.data[i] = input->data[i];
    layernorm_forward(&tb->ln1, &ln1_input);
    
    Matrix mha_out = mha_forward(&tb->mha, &ln1_input);
    
    Matrix res1 = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) {
        res1.data[i] = input->data[i] + mha_out.data[i];
    }
    
    // 2. Feed-Forward Network + Residual Connection + LayerNorm
    Matrix ln2_input = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) ln2_input.data[i] = res1.data[i];
    layernorm_forward(&tb->ln2, &ln2_input);
    
    // FFN: ReLU(x * W1 + b1) * W2 + b2
    Matrix ffn1 = matrix_multiply(&ln2_input, &tb->W1);
    // Add bias and apply ReLU
    for(int i=0; i < seq_len; i++) {
        for(int j=0; j < tb->ffn_dim; j++) {
            float val = ffn1.data[i * tb->ffn_dim + j] + tb->b1.data[j];
            ffn1.data[i * tb->ffn_dim + j] = (val > 0) ? val : 0;
        }
    }
    
    Matrix ffn2 = matrix_multiply(&ffn1, &tb->W2);
    // Add bias
    for(int i=0; i < seq_len; i++) {
        for(int j=0; j < embed_dim; j++) {
            ffn2.data[i * embed_dim + j] += tb->b2.data[j];
        }
    }
    
    Matrix res2 = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) {
        res2.data[i] = res1.data[i] + ffn2.data[i];
    }
    
    // Cleanup
    free_matrix(&ln1_input);
    free_matrix(&mha_out);
    free_matrix(&res1);
    free_matrix(&ln2_input);
    free_matrix(&ffn1);
    free_matrix(&ffn2);
    
    return res2;
}
