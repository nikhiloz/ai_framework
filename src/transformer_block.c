#include "transformer_block.h"
#include <stdlib.h>
#include <math.h>

TransformerBlock create_transformer_block(int num_heads, int embed_dim, int ffn_dim) {
    TransformerBlock tb;
    tb.embed_dim = embed_dim;
    tb.ffn_dim = ffn_dim;
    
    tb.mha = create_mha(num_heads, embed_dim);
    tb.ln1 = create_rmsnorm(embed_dim);
    tb.ln2 = create_rmsnorm(embed_dim);
    
    tb.W1 = create_matrix(embed_dim, ffn_dim);
    tb.W3 = create_matrix(embed_dim, ffn_dim); // New for SwiGLU
    tb.b1 = create_matrix(1, ffn_dim);
    tb.W2 = create_matrix(ffn_dim, embed_dim);
    tb.b2 = create_matrix(1, embed_dim);
    
    matrix_fill_random(&tb.W1);
    matrix_fill_random(&tb.W3);
    matrix_fill_random(&tb.W2);
    matrix_fill_zero(&tb.b1);
    matrix_fill_zero(&tb.b2);

    // Initialize intermediates as zero matrices (will be allocated during forward)
    tb.ln1_input = create_matrix(0, 0);
    tb.mha_out = create_matrix(0, 0);
    tb.res1 = create_matrix(0, 0);
    tb.ln2_input = create_matrix(0, 0);
    tb.ffn1 = create_matrix(0, 0);
    tb.ffn_gate = create_matrix(0, 0); // New for SwiGLU
    
    return tb;
}

void free_transformer_block(TransformerBlock *tb) {
    free_mha(&tb->mha);
    free_rmsnorm(&tb->ln1);
    free_rmsnorm(&tb->ln2);
    free_matrix(&tb->W1);
    free_matrix(&tb->W3);
    free_matrix(&tb->b1);
    free_matrix(&tb->W2);
    free_matrix(&tb->b2);
    free_matrix(&tb->ln1_input);
    free_matrix(&tb->mha_out);
    free_matrix(&tb->res1);
    free_matrix(&tb->ln2_input);
    free_matrix(&tb->ffn1);
    free_matrix(&tb->ffn_gate); // New for SwiGLU
}

Matrix transformer_block_forward(TransformerBlock *tb, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;
    int ffn_dim = tb->ffn_dim;

    // 1. MHA and Ln1 (RMSNorm) - Same
    if (tb->ln1_input.rows != seq_len) {
        free_matrix(&tb->ln1_input);
        tb->ln1_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) set_val(&tb->ln1_input, i, get_val(input, i));
    rmsnorm_forward(&tb->ln1, &tb->ln1_input);

    tb->mha_out = mha_forward(&tb->mha, &tb->ln1_input);

    if (tb->res1.rows != seq_len) {
        free_matrix(&tb->res1);
        tb->res1 = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) {
        set_val(&tb->res1, i, get_val(input, i) + get_val(&tb->mha_out, i));
    }

    // 2. FFN + RMSNorm 2
    if (tb->ln2_input.rows != seq_len) {
        free_matrix(&tb->ln2_input);
        tb->ln2_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) set_val(&tb->ln2_input, i, get_val(&tb->res1, i));
    rmsnorm_forward(&tb->ln2, &tb->ln2_input);

    // SwiGLU: (Swish(x * W1 + b1) * (x * W3)) * W2 + b2
    Matrix ffn1_raw = matrix_multiply(&tb->ln2_input, &tb->W1);
    Matrix ffn_gate_raw = matrix_multiply(&tb->ln2_input, &tb->W3);
    
    if (tb->ffn1.rows != seq_len) {
        free_matrix(&tb->ffn1);
        tb->ffn1 = create_matrix(seq_len, ffn_dim);
    }
    if (tb->ffn_gate.rows != seq_len) {
        free_matrix(&tb->ffn_gate);
        tb->ffn_gate = create_matrix(seq_len, ffn_dim);
    }
    
    for(int i=0; i < seq_len * ffn_dim; i++) {
        float x = get_val(&ffn1_raw, i) + get_val(&tb->b1, i % ffn_dim);
        // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float swish = x / (1.0f + expf(-x));
        set_val(&tb->ffn1, i, swish);
        set_val(&tb->ffn_gate, i, get_val(&ffn_gate_raw, i));
    }
    free_matrix(&ffn1_raw);
    free_matrix(&ffn_gate_raw);

    // Element-wise multiplication
    for(int i=0; i < seq_len * ffn_dim; i++) {
        set_val(&tb->ffn1, i, get_val(&tb->ffn1, i) * get_val(&tb->ffn_gate, i));
    }

    Matrix ffn2 = matrix_multiply(&tb->ffn1, &tb->W2);
    // Add bias
    for(int i=0; i < seq_len * embed_dim; i++) {
        set_val(&ffn2, i, get_val(&ffn2, i) + get_val(&tb->b2, i % embed_dim));
    }

    Matrix res2 = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) {
        set_val(&res2, i, get_val(&tb->res1, i) + get_val(&ffn2, i));
    }

    free_matrix(&ffn2);
    return res2;
}

Matrix transformer_block_backward(TransformerBlock *tb, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;
    int ffn_dim = tb->ffn_dim;
    float lr = 0.01f;

    // 1. Backward through Final Residual + FFN
    Matrix dL_dFFN2 = copy_matrix(grad_output);
    
    // Gradient through FFN bias b2
    Matrix dL_dB2 = create_matrix(1, embed_dim);
    matrix_fill_zero(&dL_dB2);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            set_val(&dL_dB2, j, get_val(&dL_dB2, j) + get_val(&dL_dFFN2, i * embed_dim + j));
        }
    }
    for (int j = 0; j < embed_dim; j++) set_val(&tb->b2, j, get_val(&tb->b2, j) - lr * get_val(&dL_dB2, j));

    // Gradient through FFN weights W2: dL/dW2 = ffn1^T * dL_dFFN2
    Matrix ffn1_T = matrix_transpose(&tb->ffn1);
    Matrix dL_dW2 = matrix_multiply(&ffn1_T, &dL_dFFN2);
    for (int i = 0; i < ffn_dim * embed_dim; i++) set_val(&tb->W2, i, get_val(&tb->W2, i) - lr * get_val(&dL_dW2, i));

    // Gradient through SwiGLU: dL/dFFN1 = dL_dFFN2 * W2^T
    Matrix W2_T = matrix_transpose(&tb->W2);
    Matrix dL_dFFN1 = matrix_multiply(&dL_dFFN2, &W2_T);

    // Gradients for SwiGLU: ffn1 = Swish(raw1) * raw3, where raw1 = x*W1+b1, raw3 = x*W3
    // dL/draw1 = dL/dffn1 * raw3 * dSwish/draw1
    // dL/draw3 = dL/dffn1 * Swish(raw1)
    Matrix dL_draw1 = create_matrix(seq_len, ffn_dim);
    Matrix dL_draw3 = create_matrix(seq_len, ffn_dim);
    
    for (int i = 0; i < seq_len * ffn_dim; i++) {
        float x = get_val(&tb->ffn1, i) / (get_val(&tb->ffn_gate, i) + 1e-9f);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float dSwish = sigmoid + x * sigmoid * (1.0f - sigmoid);
        
        set_val(&dL_draw1, i, get_val(&dL_dFFN1, i) * get_val(&tb->ffn_gate, i) * dSwish);
        set_val(&dL_draw3, i, get_val(&dL_dFFN1, i) * (x * sigmoid));
    }

    // Gradient through W1 and b1
    Matrix dL_dB1 = create_matrix(1, ffn_dim);
    matrix_fill_zero(&dL_dB1);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < ffn_dim; j++) set_val(&dL_dB1, j, get_val(&dL_dB1, j) + get_val(&dL_draw1, i * ffn_dim + j));
    }
    for (int j = 0; j < ffn_dim; j++) set_val(&tb->b1, j, get_val(&tb->b1, j) - lr * get_val(&dL_dB1, j));

    Matrix ln2_in_T = matrix_transpose(&tb->ln2_input);
    Matrix dL_dW1 = matrix_multiply(&ln2_in_T, &dL_draw1);
    for (int i = 0; i < embed_dim * ffn_dim; i++) set_val(&tb->W1, i, get_val(&tb->W1, i) - lr * get_val(&dL_dW1, i));

    // Gradient through W3
    Matrix dL_dW3 = matrix_multiply(&ln2_in_T, &dL_draw3);
    for (int i = 0; i < embed_dim * ffn_dim; i++) set_val(&tb->W3, i, get_val(&tb->W3, i) - lr * get_val(&dL_dW3, i));

    // Gradient through ln2_input
    Matrix W1_T = matrix_transpose(&tb->W1);
    Matrix W3_T = matrix_transpose(&tb->W3);
    Matrix dL_dln2_input = matrix_multiply(&dL_draw1, &W1_T);
    Matrix dL_dln2_input_gate = matrix_multiply(&dL_draw3, &W3_T);
    for (int i = 0; i < seq_len * embed_dim; i++) set_val(&dL_dln2_input, i, get_val(&dL_dln2_input, i) + get_val(&dL_dln2_input_gate, i));
    
    // RMSNorm 2 backward
    Matrix dL_dres1_ffn = rmsnorm_backward(&tb->ln2, &tb->ln2_input, &dL_dln2_input);

    // 2. Backward through First Residual + MHA
    Matrix dL_dres1 = copy_matrix(grad_output); 
    for (int i = 0; i < seq_len * embed_dim; i++) set_val(&dL_dres1, i, get_val(&dL_dres1, i) + get_val(&dL_dres1_ffn, i));

    // RMSNorm 1 backward
    Matrix dL_dln1_input = rmsnorm_backward(&tb->ln1, &tb->ln1_input, &dL_dres1);
    Matrix dL_dinput_mha = mha_backward(&tb->mha, &tb->ln1_input, &dL_dres1);

    Matrix dL_dX = create_matrix(seq_len, embed_dim);
    matrix_fill_zero(&dL_dX);
    for (int i = 0; i < seq_len * embed_dim; i++) {
        set_val(&dL_dX, i, get_val(&dL_dinput_mha, i) + get_val(&dL_dres1, i));
    }

    // Cleanup (abbreviated for brevity)
    free_matrix(&dL_dFFN2); free_matrix(&dL_dB2); free_matrix(&ffn1_T); free_matrix(&dL_dW2);
    free_matrix(&W2_T); free_matrix(&dL_dFFN1); free_matrix(&dL_draw1); free_matrix(&dL_draw3);
    free_matrix(&dL_dB1); free_matrix(&ln2_in_T); free_matrix(&dL_dW1); free_matrix(&dL_dW3);
    free_matrix(&W1_T); free_matrix(&W3_T); free_matrix(&dL_dln2_input); free_matrix(&dL_dln2_input_gate);
    free_matrix(&dL_dres1_ffn); free_matrix(&dL_dres1); free_matrix(&dL_dln1_input); free_matrix(&dL_dinput_mha);

    return dL_dX;
}
