#ifndef LM_HEAD_H
#define LM_HEAD_H

#include "matrix.h"

typedef struct {
    Matrix W; // (embed_dim, vocab_size)
    Matrix b; // (1, vocab_size)
    int embed_dim;
    int vocab_size;
} LMHead;

LMHead create_lm_head(int embed_dim, int vocab_size);
void free_lm_head(LMHead *lm);

// Forward pass: input (seq_len, embed_dim) -> output (seq_len, vocab_size)
Matrix lm_head_forward(LMHead *lm, Matrix *input);

#endif
