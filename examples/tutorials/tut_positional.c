#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include <stdio.h>

int main() {
    printf("--- Positional Encoding Test ---\n");
    
    int seq_len = 5;
    int embed_dim = 4;
    
    Matrix embeddings = create_matrix(seq_len, embed_dim);
    matrix_fill_zero(&embeddings);
    
    printf("Original Embeddings (all zero):\n");
    for(int i=0; i<seq_len; i++) {
        printf("Pos %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", get_val(&embeddings, i * embed_dim + j));
        printf("\n");
    }
    
    apply_rope(&embeddings);
    
    printf("\nEmbeddings after Positional Encoding:\n");
    for(int i=0; i<seq_len; i++) {
        printf("Pos %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", get_val(&embeddings, i * embed_dim + j));
        printf("\n");
    }
    
    free_matrix(&embeddings);
    return 0;
}
