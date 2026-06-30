#ifndef RMSNORM_H
#define RMSNORM_H

#include "matrix.h"

typedef struct {
    int dim;
    Matrix weight; // Scale parameter (gamma)
} RMSNorm;

RMSNorm create_rmsnorm(int dim);
void free_rmsnorm(RMSNorm *ln);

// Normalizes the input matrix. Modifies input in-place.
// input: Matrix of shape (seq_len, dim)
void rmsnorm_forward(RMSNorm *ln, Matrix *input);

// Backprop through RMSNorm
// input: The original input to rmsnorm_forward (seq_len, dim)
// grad_output: The gradient of the loss w.r.t. the output of rmsnorm (seq_len, dim)
// returns: Gradient w.r.t. the input (seq_len, dim)
Matrix rmsnorm_backward(RMSNorm *ln, Matrix *input, Matrix *grad_output);

#endif
