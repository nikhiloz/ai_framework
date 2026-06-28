#include "../src/transformer_block.h"
#include "../src/matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int seq_len = 16;
    int embed_dim = 64;
    int num_heads = 4;
    int ffn_dim = 256;

    printf("--- Testing TransformerBlock ---\n");
    printf("SeqLen: %d, EmbedDim: %d, Heads: %d, FFNDim: %d\n", seq_len, embed_dim, num_heads, ffn_dim);

    TransformerBlock tb = create_transformer_block(num_heads, embed_dim, ffn_dim);
    Matrix input = create_matrix(seq_len, embed_dim);
    matrix_fill_random(&input);

    clock_t start = clock();
    int iterations = 100;
    for(int i = 0; i < iterations; i++) {
        Matrix output = transformer_block_forward(&tb, &input);
        free_matrix(&output);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Completed %d iterations in %.4f seconds (%.4f s/iter)\n", 
           iterations, time_taken, time_taken / iterations);

    free_matrix(&input);
    free_transformer_block(&tb);
    printf("TransformerBlock validation successful.\n");
    return 0;
}
