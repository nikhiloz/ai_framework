#include "attention.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

AttentionLayer create_attention_layer(int embed_dim) {
    AttentionLayer al;
    al.embed_dim = embed_dim;
    
    al.K_cache = NULL;
    al.V_cache = NULL;
    al.cache_len = 0;
    
    return al;
}

AttentionLayer init_attention_layer(int embed_dim) {
    AttentionLayer al;
    al.embed_dim = embed_dim;
    
    al.K_cache = NULL;
    al.V_cache = NULL;
    al.cache_len = 0;
    return al;
}

void free_attention_layer(AttentionLayer *al) {
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

Matrix attention_forward(AttentionLayer *al, Matrix *Q, Matrix *K, Matrix *V, bool mask) {
    int seq_len = Q->rows;
    int dim = al->embed_dim;
    
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
        memcpy(new_K.data + (al->cache_len * al->K_cache->cols), K->data, seq_len * K->cols * sizeof(float));
        memcpy(new_V.data + (al->cache_len * al->V_cache->cols), V->data, seq_len * V->cols * sizeof(float));
        
        free_matrix(al->K_cache); free(al->K_cache);
        free_matrix(al->V_cache); free(al->V_cache);
        
        al->K_cache = malloc(sizeof(Matrix)); *al->K_cache = new_K;
        al->V_cache = malloc(sizeof(Matrix)); *al->V_cache = new_V;
        al->cache_len = new_cache_len;
    } else {
        // Initialize cache
        al->K_cache = malloc(sizeof(Matrix));
        *al->K_cache = copy_matrix(K);
        al->V_cache = malloc(sizeof(Matrix));
        *al->V_cache = copy_matrix(V);
        al->cache_len = seq_len;
    }
    
    // Use cached K/V for attention computation
    Matrix *used_K = al->K_cache;
    Matrix *used_V = al->V_cache;

    // 2. Compute Attention Scores
    Matrix K_T = matrix_transpose(used_K);
    Matrix scores = matrix_multiply(Q, &K_T);
    
    float scale = 1.0f / sqrtf((float)dim);
    matrix_scalar_multiply(&scores, scale);
    
    if (mask) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) set_val(&scores, i * seq_len + j, -1e9f);
        }
    }
    
    softmax_rows(&scores);
    Matrix output = matrix_multiply(&scores, used_V);
    
    free_matrix(&K_T); free_matrix(&scores);
    return output;
}
