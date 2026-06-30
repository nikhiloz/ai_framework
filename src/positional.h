#ifndef POSITIONAL_H
#define POSITIONAL_H

#include "matrix.h"

// Applies Rotary Positional Embeddings (RoPE) to an input matrix.
// input: Matrix of shape (seq_len, embed_dim)
// This function modifies the input matrix in-place by rotating the embeddings.
void apply_rope(Matrix *input);

#endif
