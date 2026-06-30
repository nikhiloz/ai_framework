#include "multihead_attention.h"
#include <stdio.h>

int main() {
    printf("--- Multi-Head Attention Test ---\n");
    
    int seq_len = 3;
    int embed_dim = 8; // Must be divisible by num_heads
    int num_heads = 2;
    
    Matrix input = create_matrix(seq_len, embed_dim);
    for(int i=0; i<seq_len * embed_dim; i++) {
        set_val(&input, i, (float)rand() / RAND_MAX);
    }
    
    printf("Input Sequence (%d x %d):\n", seq_len, embed_dim);
    for(int i=0; i<seq_len; i++) {
        printf("Token %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", get_val(&input, i * embed_dim + j));
        printf("\n");
    }
    
    MultiHeadAttention mha = create_mha(num_heads, embed_dim);
    Matrix output = mha_forward(&mha, &input);
    
    printf("\nMHA Output (%d x %d):\n", seq_len, embed_dim);
    for(int i=0; i<seq_len; i++) {
        printf("Token %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", get_val(&output, i * embed_dim + j));
        printf("\n");
    }
    
    free_matrix(&input);
    free_matrix(&output);
    free_mha(&mha);
    
    return 0;
}
