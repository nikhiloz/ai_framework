#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "transformer_block.h"
#include <stdio.h>

int main() {
    printf("--- Transformer Block Test ---\n");
    
    // Hyperparameters
    int num_heads = 2;
    int embed_dim = 8;
    int ffn_dim = 16;
    
    // 1. Text to Embeddings
    Tokenizer t = create_tokenizer(100);
    const char *text = "hello transformer";
    train_tokenizer(&t, text);
    
    int token_len;
    int *tokens = tokenize(&t, text, &token_len);
    
    Embedding e = create_embedding(t.vocab_size, embed_dim);
    Matrix x = create_matrix(token_len, embed_dim);
    embedding_lookup(&e, tokens, token_len, &x);
    
    // 2. Add Positional Encoding
    apply_positional_encoding(&x);
    
    // 3. Forward pass through one Transformer Block
    TransformerBlock tb = create_transformer_block(num_heads, embed_dim, ffn_dim);
    Matrix output = transformer_block_forward(&tb, &x);
    
    printf("Input Sequence Length: %d\n", token_len);
    printf("Output Matrix Shape: %d x %d\n", output.rows, output.cols);
    
    printf("\nFinal Output (first 2 tokens):\n");
    for(int i=0; i < (token_len < 2 ? token_len : 2); i++) {
        printf("Token %d: ", i);
        for(int j=0; j<embed_dim; j++) printf("%.4f ", output.data[i * embed_dim + j]);
        printf("\n");
    }
    
    free(tokens);
    free_matrix(&x);
    free_matrix(&output);
    free_embedding(&e);
    free_transformer_block(&tb);
    free_tokenizer(&t);
    
    return 0;
}
