#include "tokenizer.h"
#include <stdio.h>

int main() {
    printf("--- Tokenizer Test ---\n");
    
    Tokenizer t = create_tokenizer(1000);
    const char *train_text = "hello world, this is a test of the tokenizer!";
    train_tokenizer(&t, train_text);
    
    printf("Vocab size: %d\n", t.vocab_size);
    for(int i=0; i<t.vocab_size; i++) {
        printf("%d: %s\n", i, t.vocab[i]);
    }
    
    const char *test_text = "hello tokenizer";
    int len;
    int *tokens = tokenize(&t, test_text, &len);
    
    printf("\nText: %s\n", test_text);
    printf("Tokens: ");
    for(int i=0; i<len; i++) printf("%d ", tokens[i]);
    printf("\n");
    
    char *decoded = detokenize(&t, tokens, len);
    printf("Decoded: %s\n", decoded);
    
    free(tokens);
    free(decoded);
    free_tokenizer(&t);
    
    return 0;
}
