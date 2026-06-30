#include "rmsnorm.h"
#include <math.h>
#include <stdlib.h>

RMSNorm create_rmsnorm(int dim) {
    RMSNorm ln;
    ln.dim = dim;
    ln.weight = create_matrix(1, dim);
    
    // Initialize weight (gamma) to 1s
    for (int i = 0; i < dim; i++) {
        set_val(&ln.weight, i, 1.0f);
    }
    
    return ln;
}

void free_rmsnorm(RMSNorm *ln) {
    free_matrix(&ln->weight);
}

void rmsnorm_forward(RMSNorm *ln, Matrix *input) {
    int seq_len = input->rows;
    int dim = ln->dim;
    float epsilon = 1e-6f;

    for (int i = 0; i < seq_len; i++) {
        float sum_sq = 0;
        for (int j = 0; j < dim; j++) {
            float val = get_val(input, i * dim + j);
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / dim + epsilon);
        float inv_rms = 1.0f / rms;

        for (int j = 0; j < dim; j++) {
            set_val(input, i * dim + j, (get_val(input, i * dim + j) * inv_rms) * get_val(&ln->weight, j));
        }
    }
}

Matrix rmsnorm_backward(RMSNorm *ln, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int dim = ln->dim;
    float epsilon = 1e-6f;
    float lr = 0.01f;

    Matrix dL_dX = create_matrix(seq_len, dim);
    Matrix dL_dWeight = create_matrix(1, dim);
    matrix_fill_zero(&dL_dWeight);

    for (int i = 0; i < seq_len; i++) {
        float sum_sq = 0;
        for (int j = 0; j < dim; j++) {
            float val = get_val(input, i * dim + j);
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / dim + epsilon);
        float inv_rms = 1.0f / rms;

        // Gradient w.r.t weight
        for (int j = 0; j < dim; j++) {
            float normalized = get_val(input, i * dim + j) * inv_rms;
            set_val(&dL_dWeight, j, get_val(&dL_dWeight, j) + get_val(grad_output, i * dim + j) * normalized);
        }

        // Gradient w.r.t input
        float sum_grad_x_weight = 0;
        for (int j = 0; j < dim; j++) {
            sum_grad_x_weight += get_val(grad_output, i * dim + j) * get_val(&ln->weight, j) * get_val(input, i * dim + j);
        }

        for (int j = 0; j < dim; j++) {
            float term1 = get_val(grad_output, i * dim + j) * get_val(&ln->weight, j) * inv_rms;
            float term2 = (get_val(input, i * dim + j) * inv_rms * inv_rms * inv_rms * sum_grad_x_weight) / dim;
            set_val(&dL_dX, i * dim + j, term1 - term2);
        }
    }

    // Update weight
    for (int j = 0; j < dim; j++) {
        set_val(&ln->weight, j, get_val(&ln->weight, j) - lr * get_val(&dL_dWeight, j));
    }

    free_matrix(&dL_dWeight);

    return dL_dX;
}
