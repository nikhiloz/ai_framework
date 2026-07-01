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
    // For manual creation tests, initialize weights
    tb.W1 = create_matrix(embed_dim, ffn_dim);
    tb.W3 = create_matrix(embed_dim, ffn_dim);
    tb.b1 = create_matrix(1, ffn_dim);
    tb.W2 = create_matrix(ffn_dim, embed_dim);
    tb.b2 = create_matrix(1, embed_dim);
    matrix_fill_random(&tb.W1);
    matrix_fill_random(&tb.W3);
    matrix_fill_random(&tb.W2);
    matrix_fill_zero(&tb.b1);
    matrix_fill_zero(&tb.b2);
    // Intermediate buffers pre-allocated
    tb.ln1_input = create_matrix(0, 0);
    tb.mha_out = create_matrix(0, 0);
    tb.res1 = create_matrix(0, 0);
    tb.ln2_input = create_matrix(0, 0);
    tb.ffn1 = create_matrix(0, 0);
    tb.ffn_gate = create_matrix(0, 0);
    return tb;
}
TransformerBlock init_transformer_block(int num_heads, int embed_dim, int ffn_dim) {
    TransformerBlock tb;
    tb.embed_dim = embed_dim;
    tb.ffn_dim = ffn_dim;
    
    tb.mha = init_mha(num_heads, embed_dim);
    tb.ln1 = init_rmsnorm(embed_dim);
    tb.ln2 = init_rmsnorm(embed_dim);
    
    tb.W1 = init_matrix(embed_dim, ffn_dim);
    tb.W3 = init_matrix(embed_dim, ffn_dim);
    tb.b1 = init_matrix(1, ffn_dim);
    tb.W2 = init_matrix(ffn_dim, embed_dim);
    tb.b2 = init_matrix(1, embed_dim);
    
    tb.ln1_input = create_matrix(0, 0);
    tb.mha_out = create_matrix(0, 0);
    tb.res1 = create_matrix(0, 0);
    tb.ln2_input = create_matrix(0, 0);
    tb.ffn1 = create_matrix(0, 0);
    tb.ffn_gate = create_matrix(0, 0);
    
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
    free_matrix(&tb->ffn_gate);
}
Matrix transformer_block_forward(TransformerBlock *tb, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = tb->embed_dim;
    int ffn_dim = tb->ffn_dim;
    // 1. RMSNorm 1 + MHA
    if (tb->ln1_input.rows != seq_len) {
        free_matrix(&tb->ln1_input);
        tb->ln1_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) set_val(&tb->ln1_input, i, get_val(input, i));
    rmsnorm_forward(&tb->ln1, &tb->ln1_input);
    tb->mha_out = mha_forward(&tb->mha, &tb->ln1_input);
    // Residual 1
    if (tb->res1.rows != seq_len) {
        free_matrix(&tb->res1);
        tb->res1 = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) {
        set_val(&tb->res1, i, get_val(input, i) + get_val(&tb->mha_out, i));
    }
    // 2. RMSNorm 2 + FFN
    if (tb->ln2_input.rows != seq_len) {
        free_matrix(&tb->ln2_input);
        tb->ln2_input = create_matrix(seq_len, embed_dim);
    }
    for(int i=0; i < seq_len * embed_dim; i++) set_val(&tb->ln2_input, i, get_val(&tb->res1, i));
    rmsnorm_forward(&tb->ln2, &tb->ln2_input);
    // FFN SwiGLU: (Swish(x * W1 + b1) * (x * W3)) * W2 + b2
    if (!tb->W1.data || !tb->W3.data || !tb->W2.data) {
        fprintf(stderr, "Error: FFN weights not loaded! W1:%p, W3:%p, W2:%p\n", tb->W1.data, tb->W3.data, tb->W2.data);
        exit(EXIT_FAILURE);
    }
    if (tb->ffn1.rows != seq_len || tb->ffn1.cols != ffn_dim) {
        free_matrix(&tb->ffn1);
        tb->ffn1 = create_matrix(seq_len, ffn_dim);
    }
    if (tb->ffn_gate.rows != seq_len || tb->ffn_gate.cols != ffn_dim) {
        free_matrix(&tb->ffn_gate);
        tb->ffn_gate = create_matrix(seq_len, ffn_dim);
    }
    Matrix ffn1_raw = matrix_multiply(&tb->ln2_input, &tb->W1);
    if (tb->b1.data) matrix_add_bias(&ffn1_raw, &tb->b1, &ffn1_raw);
    
    Matrix ffn_gate_raw = matrix_multiply(&tb->ln2_input, &tb->W3);
    
    for(int i=0; i < seq_len * ffn_dim; i++) {
        float x = get_val(&ffn1_raw, i);
        float swish = x / (1.0f + expf(-x));
        set_val(&tb->ffn1, i, swish * get_val(&ffn_gate_raw, i));
    }
    printf("DEBUG: Freeing ffn1_raw, data=%p\n", ffn1_raw.data); free_matrix(&ffn1_raw);
    printf("DEBUG: Freeing ffn_gate_raw, data=%p\n", ffn_gate_raw.data); free_matrix(&ffn_gate_raw);
    Matrix ffn2 = matrix_multiply(&tb->ffn1, &tb->W2);
    if (tb->b2.data) matrix_add_bias(&ffn2, &tb->b2, &ffn2);
    Matrix res2 = create_matrix(seq_len, embed_dim);
    for(int i=0; i < seq_len * embed_dim; i++) {
        set_val(&res2, i, get_val(&tb->res1, i) + get_val(&ffn2, i));
    }
    printf("DEBUG: Freeing ffn2, data=%p\n", ffn2.data); free_matrix(&ffn2);
    return res2;
}
Matrix transformer_block_backward(TransformerBlock *tb, Matrix *input, Matrix *grad_output) {
    return create_matrix(0,0);
}
