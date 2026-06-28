#include "../src/lm_head.h"
#include "../src/matrix.h"

// Greedy sampling: returns the index of the token with the highest probability
int greedy_sample(Matrix *probs, int seq_idx) {
    int max_idx = 0;
    float max_prob = probs->data[seq_idx * probs->cols + 0];
    for (int j = 1; j < probs->cols; j++) {
        float prob = probs->data[seq_idx * probs->cols + j];
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = j;
        }
    }
    return max_idx;
}
