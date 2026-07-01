#include "multihead_attention.h"
#include <stdlib.h>
#include <stdbool.h>

MultiHeadAttention create_mha(int num_heads, int embed_dim) {
    MultiHeadAttention mha;
    mha.num_heads = num_heads;
    mha.embed_dim = embed_dim;
    mha.head_dim = embed_dim / num_heads;
    
    mha.heads = (AttentionLayer *)malloc(num_heads * sizeof(AttentionLayer));
    for (int i = 0; i < num_heads; i++) {
        mha.heads[i] = create_attention_layer(mha.head_dim);
    }
    
    mha.W_q = create_matrix(embed_dim, embed_dim);
    mha.W_k = create_matrix(embed_dim, embed_dim);
    mha.W_v = create_matrix(embed_dim, embed_dim);
    mha.W_o = create_matrix(embed_dim, embed_dim);
    
    matrix_fill_random(&mha.W_q);
    matrix_fill_random(&mha.W_k);
    matrix_fill_random(&mha.W_v);
    matrix_fill_random(&mha.W_o);
    
    return mha;
}

MultiHeadAttention init_mha(int num_heads, int embed_dim) {
    MultiHeadAttention mha;
    mha.num_heads = num_heads;
    mha.embed_dim = embed_dim;
    mha.head_dim = embed_dim / num_heads;
    
    mha.heads = (AttentionLayer *)malloc(num_heads * sizeof(AttentionLayer));
    for (int i = 0; i < num_heads; i++) {
        mha.heads[i] = init_attention_layer(mha.head_dim);
    }
    
    mha.W_q = init_matrix(embed_dim, embed_dim);
    mha.W_k = init_matrix(embed_dim, embed_dim);
    mha.W_v = init_matrix(embed_dim, embed_dim);
    mha.W_o = init_matrix(embed_dim, embed_dim);
    
    return mha;
}

void free_mha(MultiHeadAttention *mha) {
    for (int i = 0; i < mha->num_heads; i++) {
        free_attention_layer(&mha->heads[i]);
    }
    free(mha->heads);
    free_matrix(&mha->W_q);
    free_matrix(&mha->W_k);
    free_matrix(&mha->W_v);
    free_matrix(&mha->W_o);
}

Matrix mha_forward(MultiHeadAttention *mha, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = mha->embed_dim;
    int head_dim = mha->head_dim;
    
    printf("DEBUG: mha_forward started. seq_len: %d, embed_dim: %d\n", seq_len, embed_dim);

    // 1. Global Projections
    printf("DEBUG: Global projections...\n");
    if (!mha->W_q.data || !mha->W_k.data || !mha->W_v.data || !mha->W_o.data) {
        fprintf(stderr, "Error: MHA weights not loaded!\n");
        exit(EXIT_FAILURE);
    }
    Matrix Q_all = matrix_multiply(input, &mha->W_q);
    Matrix K_all = matrix_multiply(input, &mha->W_k);
    Matrix V_all = matrix_multiply(input, &mha->W_v);
    
    Matrix *head_outputs = (Matrix *)malloc(mha->num_heads * sizeof(Matrix));
    
    for (int h = 0; h < mha->num_heads; h++) {
        printf("DEBUG: Processing head %d...\n", h);
        Matrix q_h = create_matrix(seq_len, head_dim);
        Matrix k_h = create_matrix(seq_len, head_dim);
        Matrix v_h = create_matrix(seq_len, head_dim);
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&q_h, i * head_dim + j, get_val(&Q_all, i * embed_dim + (h * head_dim + j)));
                set_val(&k_h, i * head_dim + j, get_val(&K_all, i * embed_dim + (h * head_dim + j)));
                set_val(&v_h, i * head_dim + j, get_val(&V_all, i * embed_dim + (h * head_dim + j)));
            }
        }
        head_outputs[h] = attention_forward(&mha->heads[h], &q_h, &k_h, &v_h, true);
        free_matrix(&q_h);
        free_matrix(&k_h);
        free_matrix(&v_h);
    }
    
    printf("DEBUG: Concatenating head outputs...\n");
    Matrix concat = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < mha->num_heads; h++) {
            Matrix *head_out = &head_outputs[h];
            if (!head_out->data) {
                fprintf(stderr, "DEBUG: NULL data in head_output h=%d\n", h);
                exit(EXIT_FAILURE);
            }
            if (head_out->rows != seq_len || head_out->cols != head_dim) {
                fprintf(stderr, "DEBUG: Invalid head shape h=%d (%dx%d), expected (%dx%d)\n", h, head_out->rows, head_out->cols, seq_len, head_dim);
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < head_dim; j++) {
                int src_idx = i * head_dim + j;
                int dst_idx = i * embed_dim + (h * head_dim + j);
                
                if (src_idx < 0 || src_idx >= seq_len * head_dim) {
                    fprintf(stderr, "DEBUG: src_idx out of bounds i=%d, h=%d, j=%d, idx=%d\n", i, h, j, src_idx);
                    exit(EXIT_FAILURE);
                }
                if (dst_idx < 0 || dst_idx >= seq_len * embed_dim) {
                    fprintf(stderr, "DEBUG: dst_idx out of bounds i=%d, h=%d, j=%d, idx=%d\n", i, h, j, dst_idx);
                    exit(EXIT_FAILURE);
                }
                
                set_val(&concat, dst_idx, get_val(head_out, src_idx));
            }
        }
    }
    
    printf("DEBUG: Final projection...\n");
    Matrix output = matrix_multiply(&concat, &mha->W_o);
    
    for (int h = 0; h < mha->num_heads; h++) free_matrix(&head_outputs[h]);
    free(head_outputs);
    free_matrix(&Q_all);
    free_matrix(&K_all);
    free_matrix(&V_all);
    free_matrix(&concat);
    
    printf("DEBUG: mha_forward complete.\n");
    return output;
}

Matrix mha_backward(MultiHeadAttention *mha, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int embed_dim = mha->embed_dim;
    int head_dim = mha->head_dim;
    float lr = 0.01f;

    Matrix Wo_T = matrix_transpose(&mha->W_o);
    Matrix dL_dConcat = matrix_multiply(grad_output, &Wo_T);

    Matrix *head_outputs = (Matrix *)malloc(mha->num_heads * sizeof(Matrix));
    for (int h = 0; h < mha->num_heads; h++) {
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&head_input, i * head_dim + j, get_val(input, i * embed_dim + (h * head_dim + j)));
            }
        }
        // In a real backward pass, we would recompute Q, K, V here.
        // For now, we use dummy values or a simplified call to allow compilation.
        Matrix q_dummy = create_matrix(seq_len, head_dim);
        Matrix k_dummy = create_matrix(seq_len, head_dim);
        Matrix v_dummy = create_matrix(seq_len, head_dim);
        head_outputs[h] = attention_forward(&mha->heads[h], &q_dummy, &k_dummy, &v_dummy, true);
        free_matrix(&q_dummy);
        free_matrix(&k_dummy);
        free_matrix(&v_dummy);
        free_matrix(&head_input);
    }
    Matrix concat = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < mha->num_heads; h++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&concat, i * embed_dim + (h * head_dim + j), get_val(&head_outputs[h], i * head_dim + j));
            }
        }
    }
    Matrix concat_T = matrix_transpose(&concat);
    Matrix dL_dW_o = matrix_multiply(&concat_T, grad_output);
    for (int i = 0; i < embed_dim * embed_dim; i++) {
        set_val(&mha->W_o, i, get_val(&mha->W_o, i) - lr * get_val(&dL_dW_o, i));
    }

    Matrix dL_dX = create_matrix(seq_len, embed_dim);
    matrix_fill_zero(&dL_dX);

    for (int h = 0; h < mha->num_heads; h++) {
        Matrix dL_dHeadOut = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&dL_dHeadOut, i * head_dim + j, get_val(&dL_dConcat, i * embed_dim + (h * head_dim + j)));
            }
        }
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&head_input, i * head_dim + j, get_val(input, i * embed_dim + (h * head_dim + j)));
            }
        }
        // Matrix dL_dHeadIn = attention_backward(&mha->heads[h], &head_input, &dL_dHeadOut, true);
        Matrix dL_dHeadIn = create_matrix(seq_len, head_dim);
        matrix_fill_zero(&dL_dHeadIn);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                set_val(&dL_dX, i * embed_dim + (h * head_dim + j), get_val(&dL_dX, i * embed_dim + (h * head_dim + j)) + get_val(&dL_dHeadIn, i * head_dim + j));
            }
        }
        free_matrix(&dL_dHeadOut);
        free_matrix(&head_input);
        free_matrix(&dL_dHeadIn);
    }

    free_matrix(&Wo_T);
    free_matrix(&dL_dConcat);
    free_matrix(&concat);
    free_matrix(&concat_T);
    free_matrix(&dL_dW_o);
    for (int h = 0; h < mha->num_heads; h++) free_matrix(&head_outputs[h]);
    free(head_outputs);

    return dL_dX;
}
