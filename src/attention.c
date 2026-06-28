#include "attention.h"
#include <math.h>
#include <stdlib.h>

AttentionLayer create_attention_layer(int embed_dim) {
    AttentionLayer al;
    al.embed_dim = embed_dim;
    al.W_q = create_matrix(embed_dim, embed_dim);
    al.W_k = create_matrix(embed_dim, embed_dim);
    al.W_v = create_matrix(embed_dim, embed_dim);
    
    matrix_fill_random(&al.W_q);
    matrix_fill_random(&al.W_k);
    matrix_fill_random(&al.W_v);
    
    return al;
}

void free_attention_layer(AttentionLayer *al) {
    free_matrix(&al->W_q);
    free_matrix(&al->W_k);
    free_matrix(&al->W_v);
}

static void softmax_rows(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        float max_val = m->data[i * m->cols];
        for (int j = 1; j < m->cols; j++) {
            if (m->data[i * m->cols + j] > max_val) max_val = m->data[i * m->cols + j];
        }
        
        float sum = 0;
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = expf(m->data[i * m->cols + j] - max_val);
            sum += m->data[i * m->cols + j];
        }
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] /= sum;
        }
    }
}

Matrix attention_forward(AttentionLayer *al, Matrix *input) {
    int seq_len = input->rows;
    int dim = al->embed_dim;
    
    // 1. Compute Q, K, V
    // Q = Input * W_q, K = Input * W_k, V = Input * W_v
    Matrix Q = matrix_multiply(input, &al->W_q);
    Matrix K = matrix_multiply(input, &al->W_k);
    Matrix V = matrix_multiply(input, &al->W_v);
    
    // 2. Compute Attention Scores: Score = (Q * K^T) / sqrt(dim)
    Matrix K_T = matrix_transpose(&K);
    Matrix scores = matrix_multiply(&Q, &K_T);
    
    float scale = 1.0f / sqrtf((float)dim);
    matrix_scalar_multiply(&scores, scale);
    
    // 3. Apply Softmax to get attention weights
    softmax_rows(&scores);
    
    // 4. Compute Output: Output = Attention * V
    Matrix output = matrix_multiply(&scores, &V);
    
    // Cleanup
    free_matrix(&Q);
    free_matrix(&K);
    free_matrix(&V);
    free_matrix(&K_T);
    free_matrix(&scores);
    
    return output;
}
