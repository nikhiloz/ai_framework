#include "attention.h"
#include <stdio.h>

int main() {
    printf("--- Self-Attention Test ---\n");
    
    int seq_len = 3;
    int embed_dim = 4;
    
    // Create a dummy input sequence
    Matrix input = create_matrix(seq_len, embed_dim);
    for(int i=0; i<seq_len * embed_dim; i++) {
        input.data[i] = (float)rand() / RAND_MAX;
    }
    
    printf("Input Sequence:\n");
    for(int i=0; i<seq_len; i++) {
        printf("Token %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", input.data[i * embed_dim + j]);
        printf("\n");
    }
    
    AttentionLayer al = create_attention_layer(embed_dim);
    Matrix output = attention_forward(&al, &input);
    
    printf("\nAttention Output:\n");
    for(int i=0; i<seq_len; i++) {
        printf("Token %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", output.data[i * embed_dim + j]);
        printf("\n");
    }
    
    free_matrix(&input);
    free_matrix(&output);
    free_attention_layer(&al);
    
    return 0;
}
