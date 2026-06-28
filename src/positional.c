#include "positional.h"
#include <math.h>

void apply_positional_encoding(Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = input->cols;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / embed_dim);
            
            if (i % 2 == 0) {
                input->data[pos * embed_dim + i] += sinf(angle);
            } else {
                input->data[pos * embed_dim + i] += cosf(angle);
            }
        }
    }
}
