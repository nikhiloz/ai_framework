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

#endif
