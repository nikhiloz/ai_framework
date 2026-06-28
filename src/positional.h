#ifndef POSITIONAL_H
#define POSITIONAL_H

#include "matrix.h"

// Applies sinusoidal positional encoding to an input matrix of embeddings
// input: Matrix of shape (seq_len, embed_dim)
// This function modifies the input matrix in-place by adding the encoding
void apply_positional_encoding(Matrix *input);

#endif
