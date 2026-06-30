#include "prediction.h"
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

PredictionHead create_prediction_head(int embed_dim, int vocab_size) {
    PredictionHead ph;
    ph.embed_dim = embed_dim;
    ph.vocab_size = vocab_size;
    ph.W_out = create_matrix(embed_dim, vocab_size);
    ph.b_out = create_matrix(1, vocab_size);
    
    matrix_fill_random(&ph.W_out);
    matrix_fill_zero(&ph.b_out);
    
    return ph;
}

void free_prediction_head(PredictionHead *ph) {
    free_matrix(&ph->W_out);
    free_matrix(&ph->b_out);
}

Matrix prediction_forward(PredictionHead *ph, Matrix *input) {
    int seq_len = input->rows;
    
    Matrix logits = matrix_multiply(input, &ph->W_out);
    
    // Add bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < ph->vocab_size; j++) {
            set_val(&logits, i * ph->vocab_size + j, get_val(&logits, i * ph->vocab_size + j) + get_val(&ph->b_out, j));
        }
    }
    
    return logits;
}

Matrix prediction_backward(PredictionHead *ph, Matrix *input, Matrix *logits, int *targets) {
    int seq_len = input->rows;
    int embed_dim = ph->embed_dim;
    int vocab_size = ph->vocab_size;
    float lr = 0.01f;

    // 1. Compute dL/dLogits (Softmax-CrossEntropy gradient)
    // dL/dz_i = Prob_i - Target_i
    Matrix dL_dLogits = create_matrix(seq_len, vocab_size);
    for (int i = 0; i < seq_len; i++) {
        // Compute softmax for the row
        float max_val = get_val(logits, i * vocab_size);
        for (int j = 1; j < vocab_size; j++) {
            float val = get_val(logits, i * vocab_size + j);
            if (val > max_val) max_val = val;
        }
        float sum = 0;
        for (int j = 0; j < vocab_size; j++) {
            float p = expf(get_val(logits, i * vocab_size + j) - max_val);
            // Store temporary probability in dL_dLogits
            set_val(&dL_dLogits, i * vocab_size + j, p);
            sum += p;
        }
        
        for (int j = 0; j < vocab_size; j++) {
            float prob = get_val(&dL_dLogits, i * vocab_size + j) / sum;
            float target = (targets[i] == j) ? 1.0f : 0.0f;
            set_val(&dL_dLogits, i * vocab_size + j, prob - target);
        }
    }

    // 2. Compute dL/dW = Input^T * dL/dLogits
    Matrix input_T = matrix_transpose(input);
    Matrix dL_dW = matrix_multiply(&input_T, &dL_dLogits);

    // 3. Compute dL/dB = sum(dL/dLogits) over sequence
    Matrix dL_dB = create_matrix(1, vocab_size);
    matrix_fill_zero(&dL_dB);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < vocab_size; j++) {
            set_val(&dL_dB, j, get_val(&dL_dB, j) + get_val(&dL_dLogits, i * vocab_size + j));
        }
    }

    // 4. Update Weights (Simple SGD)
    for (int i = 0; i < embed_dim * vocab_size; i++) {
        set_val(&ph->W_out, i, get_val(&ph->W_out, i) - lr * get_val(&dL_dW, i));
    }
    for (int j = 0; j < vocab_size; j++) {
        set_val(&ph->b_out, j, get_val(&ph->b_out, j) - lr * get_val(&dL_dB, j));
    }

    // 5. Compute dL/dX (Gradient to propagate back to Transformer Block)
    // dL/dX = dL/dLogits * W_out^T
    Matrix W_out_T = matrix_transpose(&ph->W_out);
    Matrix dL_dX = matrix_multiply(&dL_dLogits, &W_out_T);

    // Cleanup
    free_matrix(&dL_dLogits);
    free_matrix(&input_T);
    free_matrix(&dL_dW);
    free_matrix(&dL_dB);
    free_matrix(&W_out_T);

    return dL_dX;
}

float cross_entropy_loss(Matrix *logits, int *targets) {
    int seq_len = logits->rows;
    int vocab_size = logits->cols;
    float total_loss = 0;

    for (int i = 0; i < seq_len; i++) {
        // 1. Compute Softmax for the row (probability distribution)
        float max_val = get_val(logits, i * vocab_size);
        for (int j = 1; j < vocab_size; j++) {
            float val = get_val(logits, i * vocab_size + j);
            if (val > max_val) max_val = val;
        }

        float sum = 0;
        float *row_probs = (float *)malloc(vocab_size * sizeof(float));
        for (int j = 0; j < vocab_size; j++) {
            row_probs[j] = expf(get_val(logits, i * vocab_size + j) - max_val);
            sum += row_probs[j];
        }

        // 2. Loss = -log(prob of target token)
        int target = targets[i];
        if (target >= 0 && target < vocab_size) {
            float prob = row_probs[target] / sum;
            total_loss -= logf(prob + 1e-9f);
        }

        free(row_probs);
    }

    return total_loss / seq_len;
}
