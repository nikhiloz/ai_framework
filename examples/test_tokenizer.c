#include "../src/tokenizer.h"
#include <stdio.h>
#include <string.h>

int main() {
    Tokenizer t = create_tokenizer(260); // 256 bytes + 4 merges
    const char *text = "hello hello world";
    
    printf("Training tokenizer...\n");
    train_tokenizer(&t, text);
    
    printf("Vocab size: %d\n", t.vocab_size);
    printf("Number of merges learned: %d\n", t.num_merges);
    
    int out_len;
    int *tokens = tokenize(&t, text, &out_len);
    
    printf("Tokenized length: %d\n", out_len);
    char *detokenized = detokenize(&t, tokens, out_len);
    printf("Detokenized: %s\n", detokenized);
    
    free(tokens);
    free(detokenized);
    free_tokenizer(&t);
    return 0;
}
