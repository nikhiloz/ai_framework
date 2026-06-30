#include "embedding.h"
#include <stdlib.h>

Embedding create_embedding(int vocab_size, int embedding_dim) {
    Embedding e;
    e.vocab_size = vocab_size;
    e.embedding_dim = embedding_dim;
    e.weights = create_matrix(vocab_size, embedding_dim);
    matrix_fill_random(&e.weights);
    return e;
}

void free_embedding(Embedding *e) {
    free_matrix(&e->weights);
}

void embedding_lookup(Embedding *e, int *tokens, int token_len, Matrix *out_matrix) {
    // out_matrix should already be created with size (token_len, embedding_dim)
    for (int i = 0; i < token_len; i++) {
        int token_id = tokens[i];
        if (token_id < 0 || token_id >= e->vocab_size) {
            // Fill with zeros or a special <UNK> vector if available
            for (int j = 0; j < e->embedding_dim; j++) {
                set_val(out_matrix, i * e->embedding_dim + j, 0.0f);
            }
        } else {
            // Copy the row corresponding to the token_id
            for (int j = 0; j < e->embedding_dim; j++) {
                set_val(out_matrix, i * e->embedding_dim + j, 
                        get_val(&e->weights, token_id * e->embedding_dim + j));
            }
        }
    }
}
