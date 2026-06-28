#ifndef PREDICTION_H
#define PREDICTION_H

#include "matrix.h"

typedef struct {
    int embed_dim;
    int vocab_size;
    Matrix W_out; // Shape: (embed_dim, vocab_size)
    Matrix b_out; // Shape: (1, vocab_size)
} PredictionHead;

PredictionHead create_prediction_head(int embed_dim, int vocab_size);
void free_prediction_head(PredictionHead *ph);

// Maps the transformer output to vocabulary logits
// input: Matrix of shape (seq_len, embed_dim)
// returns: Matrix of shape (seq_len, vocab_size)
Matrix prediction_forward(PredictionHead *ph, Matrix *input);

// Backprop through the prediction head
// input: The original input to prediction_forward (seq_len, embed_dim)
// logits: The output of prediction_forward (seq_len, vocab_size)
// targets: Correct token IDs (seq_len)
// out_grad: The gradient with respect to the input (dL/dx) - allocated inside
Matrix prediction_backward(PredictionHead *ph, Matrix *input, Matrix *logits, int *targets);

// Computes the Cross-Entropy loss for a sequence
// logits: The output of prediction_forward (seq_len, vocab_size)
// targets: Array of correct token IDs (seq_len)
float cross_entropy_loss(Matrix *logits, int *targets);

#endif
