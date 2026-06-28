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

Matrix layernorm_backward(LayerNorm *ln, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int dim = ln->dim;
    float epsilon = 1e-5f;
    float lr = 0.01f;

    Matrix dL_dX = create_matrix(seq_len, dim);
    Matrix dL_dGamma = create_matrix(1, dim);
    Matrix dL_dBeta = create_matrix(1, dim);
    matrix_fill_zero(&dL_dGamma);
    matrix_fill_zero(&dL_dBeta);

    for (int i = 0; i < seq_len; i++) {
        float mean = 0;
        for (int j = 0; j < dim; j++) mean += input->data[i * dim + j];
        mean /= dim;

        float variance = 0;
        for (int j = 0; j < dim; j++) {
            float diff = input->data[i * dim + j] - mean;
            variance += diff * diff;
        }
        variance /= dim;
        float std_inv = 1.0f / sqrtf(variance + epsilon);

        for (int j = 0; j < dim; j++) {
            float x_hat = (input->data[i * dim + j] - mean) * std_inv;
            float dy = grad_output->data[i * dim + j];

            // Gradients for parameters
            dL_dGamma.data[j] += dy * x_hat;
            dL_dBeta.data[j] += dy;

            // Gradient for input
            // dL/dx = (gamma / std) * [dy - (sum(dy * gamma) / dim) - (sum(dy * gamma * x_hat) / dim) * x_hat]
            float sum_dy_gamma = 0;
            for (int k = 0; k < dim; k++) {
                sum_dy_gamma += grad_output->data[i * dim + k] * ln->gamma.data[k];
            }
            
            float sum_dy_gamma_xhat = 0;
            for (int k = 0; k < dim; k++) {
                float x_hat_k = (input->data[i * dim + k] - mean) * std_inv;
                sum_dy_gamma_xhat += grad_output->data[i * dim + k] * ln->gamma.data[k] * x_hat_k;
            }

            dL_dX.data[i * dim + j] = (ln->gamma.data[j] * std_inv) * 
                (dy - (sum_dy_gamma / dim) - (sum_dy_gamma_xhat / dim) * x_hat);
        }
    }

    // Update parameters
    for (int j = 0; j < dim; j++) {
        ln->gamma.data[j] -= lr * dL_dGamma.data[j];
        ln->beta.data[j] -= lr * dL_dBeta.data[j];
    }

    free_matrix(&dL_dGamma);
    free_matrix(&dL_dBeta);

    return dL_dX;
}
