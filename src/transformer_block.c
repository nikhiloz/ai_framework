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

    // Initialize intermediates as zero matrices (will be allocated during forward)
    tb.ln1_input = create_matrix(0, 0);
    tb.mha_out = create_matrix(0, 0);
    tb.res1 = create_matrix(0, 0);
    tb.ln2_input = create_matrix(0, 0);
    tb.ffn1 = create_matrix(0, 0);
    
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
    free_matrix(&tb->ln1_input);
    free_matrix(&tb->mha_out);
    free_matrix(&tb->res1);
    free_matrix(&tb->ln2_input);
    free_matrix(&tb->ffn1);
}
Matrix transformer_block_forward(TransformerBlock *tb, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;

    // 1. Multi-Head Attention + Residual Connection + LayerNorm
    if (tb->ln1_input.rows != seq_len) {
        free_matrix(&tb->ln1_input);
        tb->ln1_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) tb->ln1_input.data[i] = input->data[i];
    layernorm_forward(&tb->ln1, &tb->ln1_input);

    tb->mha_out = mha_forward(&tb->mha, &tb->ln1_input);

    if (tb->res1.rows != seq_len) {
        free_matrix(&tb->res1);
        tb->res1 = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) {
        tb->res1.data[i] = input->data[i] + tb->mha_out.data[i];
    }

    // 2. Feed-Forward Network + Residual Connection + LayerNorm
    if (tb->ln2_input.rows != seq_len) {
        free_matrix(&tb->ln2_input);
        tb->ln2_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) tb->ln2_input.data[i] = tb->res1.data[i];
    layernorm_forward(&tb->ln2, &tb->ln2_input);

    // FFN: ReLU(x * W1 + b1) * W2 + b2
    Matrix ffn1_raw = matrix_multiply(&tb->ln2_input, &tb->W1);
    if (tb->ffn1.rows != seq_len) {
        free_matrix(&tb->ffn1);
        tb->ffn1 = create_matrix(seq_len, tb->ffn_dim);
    }
    for(int i=0; i < seq_len; i++) {
        for(int j=0; j < tb->ffn_dim; j++) {
            float val = ffn1_raw.data[i * tb->ffn_dim + j] + tb->b1.data[j];
            tb->ffn1.data[i * tb->ffn_dim + j] = (val > 0) ? val : 0;
        }
    }
    free_matrix(&ffn1_raw);

    Matrix ffn2 = matrix_multiply(&tb->ffn1, &tb->W2);
    // Add bias
    for(int i=0; i < seq_len; i++) {
        for(int j=0; j < embed_dim; j++) {
            ffn2.data[i * embed_dim + j] += tb->b2.data[j];
        }
    }

    Matrix res2 = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) {
        res2.data[i] = tb->res1.data[i] + ffn2.data[i];
    }

    // Cleanup only temporary results
    free_matrix(&ffn2);

    return res2;
}

Matrix transformer_block_backward(TransformerBlock *tb, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;
    int ffn_dim = tb->ffn_dim;
    float lr = 0.01f;

    // 1. Backward through Final Residual + FFN
    // grad_output is dL/dRes2. dL/dRes2 = dL/dRes1 + dL/dFFN2
    Matrix dL_dFFN2 = copy_matrix(grad_output);
    
    // Gradient through FFN bias b2
    Matrix dL_dB2 = create_matrix(1, embed_dim);
    matrix_fill_zero(&dL_dB2);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            dL_dB2.data[j] += dL_dFFN2.data[i * embed_dim + j];
        }
    }
    for (int j = 0; j < embed_dim; j++) tb->b2.data[j] -= lr * dL_dB2.data[j];

    // Gradient through FFN weights W2: dL/dW2 = FFN1^T * dL/dFFN2
    Matrix ffn1_T = matrix_transpose(&tb->ffn1);
    Matrix dL_dW2 = matrix_multiply(&ffn1_T, &dL_dFFN2);
    for (int i = 0; i < ffn_dim * embed_dim; i++) tb->W2.data[i] -= lr * dL_dW2.data[i];

    // Gradient through FFN1: dL/dFFN1 = dL/dFFN2 * W2^T
    Matrix W2_T = matrix_transpose(&tb->W2);
    Matrix dL_dFFN1 = matrix_multiply(&dL_dFFN2, &W2_T);

    // Gradient through ReLU and b1
    Matrix dL_dB1 = create_matrix(1, ffn_dim);
    matrix_fill_zero(&dL_dB1);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < ffn_dim; j++) {
            float val = tb->ffn1.data[i * ffn_dim + j];
            float grad = (val > 0) ? dL_dFFN1.data[i * ffn_dim + j] : 0;
            dL_dB1.data[j] += grad;
            dL_dFFN1.data[i * ffn_dim + j] = grad; // Update in-place for W1 grad
        }
    }
    for (int j = 0; j < ffn_dim; j++) tb->b1.data[j] -= lr * dL_dB1.data[j];

    // Gradient through FFN weights W1: dL/dW1 = ln2_input^T * dL/dFFN1
    Matrix ln2_in_T = matrix_transpose(&tb->ln2_input);
    Matrix dL_dW1 = matrix_multiply(&ln2_in_T, &dL_dFFN1);
    for (int i = 0; i < embed_dim * ffn_dim; i++) tb->W1.data[i] -= lr * dL_dW1.data[i];

    // Gradient through ln2_input
    Matrix W1_T = matrix_transpose(&tb->W1);
    Matrix dL_dln2_input = matrix_multiply(&dL_dFFN1, &W1_T);
    
    // LayerNorm 2 backward
    Matrix dL_dres1_ffn = layernorm_backward(&tb->ln2, &tb->ln2_input, &dL_dln2_input);

    // 2. Backward through First Residual + MHA
    // dL/dRes1 = dL/dln2_input (from FFN path) + dL/dMHA_out (from MHA path)
    // But wait, res1 is the sum of input and mha_out.
    // So grad_output to the first residual block is just the gradient from the FFN path.
    Matrix dL_dres1 = copy_matrix(grad_output); 
    // Add the gradient from the FFN part
    for (int i = 0; i < seq_len * embed_dim; i++) {
        dL_dres1.data[i] += dL_dres1_ffn.data[i];
    }

    // LayerNorm 1 backward
    Matrix dL_dln1_input = layernorm_backward(&tb->ln1, &tb->ln1_input, &dL_dres1);
    
    // MHA backward
    Matrix dL_dinput_mha = mha_backward(&tb->mha, &tb->ln1_input, &dL_dres1);

    // Final gradient: dL/dX = grad from MHA path + grad from Residual path
    Matrix dL_dX = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len * embed_dim; i++) {
        dL_dX.data[i] = dL_dinput_mha.data[i] + dL_dres1.data[i];
    }

    // Cleanup
    free_matrix(&dL_dFFN2);
    free_matrix(&dL_dB2);
    free_matrix(&ffn1_T);
    free_matrix(&dL_dW2);
    free_matrix(&W2_T);
    free_matrix(&dL_dFFN1);
    free_matrix(&dL_dB1);
    free_matrix(&ln2_in_T);
    free_matrix(&dL_dW1);
    free_matrix(&W1_T);
    free_matrix(&dL_dln2_input);
    free_matrix(&dL_dres1_ffn);
    free_matrix(&dL_dres1);
    free_matrix(&dL_dln1_input);
    free_matrix(&dL_dinput_mha);

    return dL_dX;
}
