#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "matrix.h"

typedef struct {
    int dim;
    Matrix gamma; // Scale parameter
    Matrix beta;  // Shift parameter
} LayerNorm;

LayerNorm create_layernorm(int dim);
void free_layernorm(LayerNorm *ln);

// Normalizes the input matrix. Modifies input in-place.
// input: Matrix of shape (seq_len, dim)
void layernorm_forward(LayerNorm *ln, Matrix *input);

// Backprop through LayerNorm
// input: The original input to layernorm_forward (seq_len, dim)
// grad_output: The gradient of the loss w.r.t. the output of layernorm (seq_len, dim)
// returns: Gradient w.r.t. the input (seq_len, dim)
Matrix layernorm_backward(LayerNorm *ln, Matrix *input, Matrix *grad_output);

#endif
