#include "layernorm.h"
#include <math.h>
#include <stdlib.h>

LayerNorm create_layernorm(int dim) {
    LayerNorm ln;
    ln.dim = dim;
    ln.gamma = create_matrix(1, dim);
    ln.beta = create_matrix(1, dim);
    
    // Initialize gamma to 1s and beta to 0s
    for (int i = 0; i < dim; i++) {
        ln.gamma.data[i] = 1.0f;
        ln.beta.data[i] = 0.0f;
    }
    
    return ln;
}

void free_layernorm(LayerNorm *ln) {
    free_matrix(&ln->gamma);
    free_matrix(&ln->beta);
}

void layernorm_forward(LayerNorm *ln, Matrix *input) {
    int seq_len = input->rows;
    int dim = ln->dim;
    float epsilon = 1e-5f;

    for (int i = 0; i < seq_len; i++) {
        float mean = 0;
        for (int j = 0; j < dim; j++) {
            mean += input->data[i * dim + j];
        }
        mean /= dim;

        float variance = 0;
        for (int j = 0; j < dim; j++) {
            float diff = input->data[i * dim + j] - mean;
            variance += diff * diff;
        }
        variance /= dim;

        float std_inv = 1.0f / sqrtf(variance + epsilon);

        for (int j = 0; j < dim; j++) {
            float normalized = (input->data[i * dim + j] - mean) * std_inv;
            input->data[i * dim + j] = normalized * ln->gamma.data[j] + ln->beta.data[j];
        }
    }
}
