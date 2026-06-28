#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int pair[2];
    int replacement;
} MergeRule;

typedef struct {
    char **vocab;          // Token string to ID mapping
    int vocab_size;
    int max_vocab_size;
    
    MergeRule *merges;
    int num_merges;
} Tokenizer;

Tokenizer create_tokenizer(int max_vocab_size);
void free_tokenizer(Tokenizer *t);

// Training
void train_tokenizer(Tokenizer *t, const char *text);

// Usage
int* tokenize(Tokenizer *t, const char *text, int *out_len);
char* detokenize(Tokenizer *t, int *tokens, int len);

#endif
