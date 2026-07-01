#include "positional.h"
#include <math.h>

void apply_rope(Matrix *input) {
    printf("DEBUG: apply_rope start\n");
    fflush(stdout);
    int seq_len = input->rows;
    int embed_dim = input->cols;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < embed_dim; i += 2) {
            float theta = pos / powf(10000.0f, (float)i / embed_dim);
            
            float x1 = get_val(input, pos * embed_dim + i);
            float x2 = get_val(input, pos * embed_dim + i + 1);

            // RoPE rotation:
            // x1' = x1 * cos(theta) - x2 * sin(theta)
            // x2' = x1 * sin(theta) + x2 * cos(theta)
            set_val(input, pos * embed_dim + i, x1 * cosf(theta) - x2 * sinf(theta));
            set_val(input, pos * embed_dim + i + 1, x1 * sinf(theta) + x2 * cosf(theta));
        }
    }
    printf("DEBUG: apply_rope end\n");
    fflush(stdout);
}
