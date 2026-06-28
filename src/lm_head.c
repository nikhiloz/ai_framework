#include "lm_head.h"
#include <math.h>
#include <stdlib.h>

LMHead create_lm_head(int embed_dim, int vocab_size) {
    LMHead lm;
    lm.embed_dim = embed_dim;
    lm.vocab_size = vocab_size;
    lm.W = create_matrix(embed_dim, vocab_size);
    lm.b = create_matrix(1, vocab_size);
    
    matrix_fill_random(&lm.W);
    matrix_fill_zero(&lm.b);
    
    return lm;
}

void free_lm_head(LMHead *lm) {
    free_matrix(&lm->W);
    free_matrix(&lm->b);
}

static void softmax(Matrix *m) {
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

Matrix lm_head_forward(LMHead *lm, Matrix *input) {
    Matrix logits = matrix_multiply(input, &lm->W);
    // Add bias
    for (int i = 0; i < logits.rows; i++) {
        for (int j = 0; j < logits.cols; j++) {
            logits.data[i * logits.cols + j] += lm->b.data[j];
        }
    }
    
    softmax(&logits);
    return logits;
}
