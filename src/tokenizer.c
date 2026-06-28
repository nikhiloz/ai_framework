#include "tokenizer.h"
#include <stdbool.h>

Tokenizer create_tokenizer(int max_vocab_size) {
    Tokenizer t;
    t.vocab_size = 0;
    t.max_vocab_size = max_vocab_size;
    t.vocab = (char **)malloc(max_vocab_size * sizeof(char *));
    t.merges = NULL;
    t.num_merges = 0;
    return t;
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    if (t->merges) free(t->merges);
}

static int get_token_id(Tokenizer *t, const char *s) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (strcmp(t->vocab[i], s) == 0) return i;
    }
    return -1;
}

static void add_to_vocab(Tokenizer *t, const char *s) {
    if (get_token_id(t, s) != -1) return;
    if (t->vocab_size >= t->max_vocab_size) return;
    
    t->vocab[t->vocab_size] = strdup(s);
    t->vocab_size++;
}

void train_tokenizer(Tokenizer *t, const char *text) {
    // Initial vocab: all unique characters
    for (int i = 0; text[i] != '\0'; i++) {
        char s[2] = {text[i], '\0'};
        add_to_vocab(t, s);
    }

    // Basic BPE training loop (Simplified for prototype)
    // In a real BPE, we would count pairs and merge the most frequent
    // For this prototype, we'll implement a basic character-level fallback
    // and allow for manual merge additions or a simple greedy merge.
    
    // Placeholder for complex BPE logic. 
    // For now, this tokenizer starts as a character-level tokenizer.
}

int* tokenize(Tokenizer *t, const char *text, int *out_len) {
    int len = strlen(text);
    int *tokens = (int *)malloc(len * sizeof(int));
    int count = 0;
    
    for (int i = 0; i < len; i++) {
        char s[2] = {text[i], '\0'};
        int id = get_token_id(t, s);
        if (id != -1) {
            tokens[count++] = id;
        } else {
            tokens[count++] = 0; // Unknown token
        }
    }
    
    *out_len = count;
    return tokens;
}

char* detokenize(Tokenizer *t, int *tokens, int len) {
    // Calculate total length
    int total_len = 0;
    for (int i = 0; i < len; i++) {
        if (tokens[i] >= 0 && tokens[i] < t->vocab_size) {
            total_len += strlen(t->vocab[tokens[i]]);
        }
    }
    
    char *result = (char *)malloc(total_len + 1);
    result[0] = '\0';
    for (int i = 0; i < len; i++) {
        if (tokens[i] >= 0 && tokens[i] < t->vocab_size) {
            strcat(result, t->vocab[tokens[i]]);
        }
    }
    return result;
}
