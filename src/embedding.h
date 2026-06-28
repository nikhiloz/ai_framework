#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "matrix.h"

typedef struct {
    int vocab_size;
    int embedding_dim;
    Matrix weights; // Shape: (vocab_size, embedding_dim)
} Embedding;

Embedding create_embedding(int vocab_size, int embedding_dim);
void free_embedding(Embedding *e);

// Maps a sequence of tokens to a matrix of embeddings
// tokens: array of token IDs
// out_matrix: pointer to a matrix that will be filled (rows = token_len, cols = embedding_dim)
void embedding_lookup(Embedding *e, int *tokens, int token_len, Matrix *out_matrix);

#endif
