#include "tokenizer.h"
#include "embedding.h"
#include <stdio.h>

int main() {
    printf("--- Embedding Test ---\n");
    
    // 1. Setup Tokenizer
    Tokenizer t = create_tokenizer(100);
    const char *text = "hello ai";
    train_tokenizer(&t, text);
    
    int token_len;
    int *tokens = tokenize(&t, text, &token_len);
    
    printf("Text: %s\n", text);
    printf("Tokens: ");
    for(int i=0; i<token_len; i++) printf("%d ", tokens[i]);
    printf("\n");
    
    // 2. Setup Embeddings
    int embed_dim = 4;
    Embedding e = create_embedding(t.vocab_size, embed_dim);
    
    Matrix embeddings = create_matrix(token_len, embed_dim);
    embedding_lookup(&e, tokens, token_len, &embeddings);
    
    printf("\nEmbeddings Matrix (%d x %d):\n", token_len, embed_dim);
    for(int i=0; i<token_len; i++) {
        printf("Token %d: ", tokens[i]);
        for(int j=0; j<embed_dim; j++) {
            printf("%.4f ", get_val(&embeddings, i * embed_dim + j));
        }
        printf("\n");
    }
    
    free(tokens);
    free_matrix(&embeddings);
    free_embedding(&e);
    free_tokenizer(&t);
    
    return 0;
}
