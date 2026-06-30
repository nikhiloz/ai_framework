#include "attention.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

AttentionLayer create_attention_layer(int embed_dim) {
    AttentionLayer al;
    al.embed_dim = embed_dim;
    al.W_q = create_matrix(embed_dim, embed_dim);
    al.W_k = create_matrix(embed_dim, embed_dim);
    al.W_v = create_matrix(embed_dim, embed_dim);
    
    matrix_fill_random(&al.W_q);
    matrix_fill_random(&al.W_k);
    matrix_fill_random(&al.W_v);
    
    al.K_cache = NULL;
    al.V_cache = NULL;
    al.cache_len = 0;
    
    return al;
}

void free_attention_layer(AttentionLayer *al) {
    free_matrix(&al->W_q);
    free_matrix(&al->W_k);
    free_matrix(&al->W_v);
    
    if (al->K_cache) {
        free_matrix(al->K_cache);
        free(al->K_cache);
    }
    if (al->V_cache) {
        free_matrix(al->V_cache);
        free(al->V_cache);
    }
}

static void softmax_rows(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        float max_val = get_val(m, i * m->cols);
        for (int j = 1; j < m->cols; j++) {
            float val = get_val(m, i * m->cols + j);
            if (val > max_val) max_val = val;
        }
        
        float sum = 0;
        for (int j = 0; j < m->cols; j++) {
            float val = expf(get_val(m, i * m->cols + j) - max_val);
            set_val(m, i * m->cols + j, val);
            sum += val;
        }
        for (int j = 0; j < m->cols; j++) {
            set_val(m, i * m->cols + j, get_val(m, i * m->cols + j) / sum);
        }
    }
}

Matrix attention_forward(AttentionLayer *al, Matrix *input, bool mask) {
    int seq_len = input->rows;
    int dim = al->embed_dim;
    
    // 1. Compute Q, K, V
    Matrix Q = matrix_multiply(input, &al->W_q);
    Matrix K = matrix_multiply(input, &al->W_k);
    Matrix V = matrix_multiply(input, &al->W_v);
    
    // KV Cache management
    if (al->K_cache != NULL) {
        // Append current K, V to cache
        int new_cache_len = al->cache_len + seq_len;
        Matrix new_K = create_matrix(new_cache_len, al->K_cache->cols);
        Matrix new_V = create_matrix(new_cache_len, al->V_cache->cols);
        
        // Copy old cache
        memcpy(new_K.data, al->K_cache->data, al->cache_len * al->K_cache->cols * sizeof(float));
        memcpy(new_V.data, al->V_cache->data, al->cache_len * al->V_cache->cols * sizeof(float));
        
        // Append new K, V
        memcpy(new_K.data + (al->cache_len * al->K_cache->cols), K.data, seq_len * K.cols * sizeof(float));
        memcpy(new_V.data + (al->cache_len * al->V_cache->cols), V.data, seq_len * V.cols * sizeof(float));
        
        free_matrix(al->K_cache); free(al->K_cache);
        free_matrix(al->V_cache); free(al->V_cache);
        
        al->K_cache = malloc(sizeof(Matrix)); *al->K_cache = new_K;
        al->V_cache = malloc(sizeof(Matrix)); *al->V_cache = new_V;
        al->cache_len = new_cache_len;
    } else {
        // Initialize cache
        al->K_cache = malloc(sizeof(Matrix));
        *al->K_cache = copy_matrix(&K);
        al->V_cache = malloc(sizeof(Matrix));
        *al->V_cache = copy_matrix(&V);
        al->cache_len = seq_len;
    }
    
    // Use cached K/V for attention computation
    Matrix *used_K = al->K_cache;
    Matrix *used_V = al->V_cache;

    // 2. Compute Attention Scores
    Matrix K_T = matrix_transpose(used_K);
    Matrix scores = matrix_multiply(&Q, &K_T);
    
    float scale = 1.0f / sqrtf((float)dim);
    matrix_scalar_multiply(&scores, scale);
    
    if (mask) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) set_val(&scores, i * seq_len + j, -1e9f);
        }
    }
    
    softmax_rows(&scores);
    Matrix output = matrix_multiply(&scores, used_V);
    
    free_matrix(&Q); free_matrix(&K); free_matrix(&V); free_matrix(&K_T); free_matrix(&scores);
    return output;
}

Matrix attention_backward(AttentionLayer *al, Matrix *input, Matrix *grad_output, bool mask) {
    int seq_len = input->rows;
    int dim = al->embed_dim;
    float lr = 0.01f;

    Matrix Q = matrix_multiply(input, &al->W_q);
    Matrix K = matrix_multiply(input, &al->W_k);
    Matrix V = matrix_multiply(input, &al->W_v);
    Matrix K_T = matrix_transpose(&K);
    Matrix scores = matrix_multiply(&Q, &K_T);
    float scale = 1.0f / sqrtf((float)dim);
    matrix_scalar_multiply(&scores, scale);
    if (mask) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) set_val(&scores, i * seq_len + j, -1e9f);
        }
    }
    
    Matrix S = create_matrix(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++) {
        float max_val = get_val(&scores, i * seq_len);
        for (int j = 1; j < seq_len; j++) {
            float val = get_val(&scores, i * seq_len + j);
            if (val > max_val) max_val = val;
        }
        float sum = 0;
        for (int j = 0; j < seq_len; j++) {
            float val = expf(get_val(&scores, i * seq_len + j) - max_val);
            set_val(&S, i * seq_len + j, val);
            sum += val;
        }
        for (int j = 0; j < seq_len; j++) {
            set_val(&S, i * seq_len + j, get_val(&S, i * seq_len + j) / sum);
        }
    }

    Matrix S_T = matrix_transpose(&S);
    Matrix dL_dV = matrix_multiply(&S_T, grad_output);
    Matrix V_T = matrix_transpose(&V);
    Matrix dL_dS = matrix_multiply(grad_output, &V_T);

    Matrix dL_dScores = create_matrix(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0;
            for (int k = 0; k < seq_len; k++) {
                float s_ik = get_val(&S, i * seq_len + k);
                float diff = (j == k) ? 1.0f : 0.0f;
                sum += s_ik * (diff - get_val(&S, i * seq_len + j)) * get_val(&dL_dS, i * seq_len + k);
            }
            set_val(&dL_dScores, i * seq_len + j, sum);
        }
    }

    Matrix dL_dScores_scaled = create_matrix(seq_len, seq_len);
    for (int i = 0; i < seq_len * seq_len; i++) {
        set_val(&dL_dScores_scaled, i, get_val(&dL_dScores, i) * scale);
    }

    Matrix dL_dQ = matrix_multiply(&dL_dScores_scaled, &K);
    Matrix dL_dScores_T = matrix_transpose(&dL_dScores_scaled);
    Matrix dL_dK = matrix_multiply(&dL_dScores_T, &Q);

    Matrix Q_T = matrix_transpose(&Q);
    Matrix dL_dWq = matrix_multiply(&Q_T, &dL_dQ);
    Matrix K_T2 = matrix_transpose(&K);
    Matrix dL_dWk = matrix_multiply(&K_T2, &dL_dK);
    Matrix V_T2 = matrix_transpose(&V);
    Matrix dL_dWv = matrix_multiply(&V_T2, &dL_dV);

    for (int i = 0; i < dim * dim; i++) {
        set_val(&al->W_q, i, get_val(&al->W_q, i) - lr * get_val(&dL_dWq, i));
        set_val(&al->W_k, i, get_val(&al->W_k, i) - lr * get_val(&dL_dWk, i));
        set_val(&al->W_v, i, get_val(&al->W_v, i) - lr * get_val(&dL_dWv, i));
    }

    Matrix Wq_T = matrix_transpose(&al->W_q);
    Matrix Wk_T = matrix_transpose(&al->W_k);
    Matrix Wv_T = matrix_transpose(&al->W_v);
    Matrix grad_q = matrix_multiply(&dL_dQ, &Wq_T);
    Matrix grad_k = matrix_multiply(&dL_dK, &Wk_T);
    Matrix grad_v = matrix_multiply(&dL_dV, &Wv_T);
    
    Matrix dL_dX = create_matrix(seq_len, dim);
    matrix_fill_zero(&dL_dX);
    for (int i = 0; i < seq_len * dim; i++) {
        set_val(&dL_dX, i, get_val(&grad_q, i) + get_val(&grad_k, i) + get_val(&grad_v, i));
    }

    free_matrix(&Q); free_matrix(&K); free_matrix(&V); free_matrix(&K_T); free_matrix(&scores);
    free_matrix(&S); free_matrix(&S_T); free_matrix(&dL_dV); free_matrix(&V_T); free_matrix(&dL_dS);
    free_matrix(&dL_dScores); free_matrix(&dL_dScores_scaled); free_matrix(&dL_dQ); free_matrix(&dL_dK);
    free_matrix(&Q_T); free_matrix(&dL_dWq); free_matrix(&K_T2); free_matrix(&dL_dWk); free_matrix(&V_T2); free_matrix(&dL_dWv);
    free_matrix(&Wq_T); free_matrix(&Wk_T); free_matrix(&Wv_T); free_matrix(&grad_q); free_matrix(&grad_k);
    free_matrix(&grad_v);

    return dL_dX;
}
