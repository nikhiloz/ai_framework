#include "tokenizer.h"
#include <stdbool.h>
#include <string.h>

Tokenizer create_tokenizer(int max_vocab_size) {
    Tokenizer t;
    t.vocab_size = 0;
    t.max_vocab_size = max_vocab_size;
    t.vocab = (char **)malloc(max_vocab_size * sizeof(char *));
    t.merges = (MergeRule *)malloc((max_vocab_size - 256) * sizeof(MergeRule)); // Upper bound
    t.num_merges = 0;
    return t;
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->merges);
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

// Helper to count pairs in current token sequence
typedef struct { int pair[2]; int count; } PairCount;

void train_tokenizer(Tokenizer *t, const char *text) {
    // 1. Initial vocab: all unique bytes/characters
    for (int i = 0; i < 256; i++) {
        char s[2] = {(char)i, '\0'};
        add_to_vocab(t, s);
    }

    // 2. Initial sequence
    int len = strlen(text);
    int *seq = (int *)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) seq[i] = (unsigned char)text[i];
    int seq_len = len;

    // 3. Iterative BPE training
    while (t->vocab_size < t->max_vocab_size) {
        // Count pairs
        int num_pairs = 0;
        PairCount *pairs = (PairCount *)malloc(seq_len * sizeof(PairCount));
        for (int i = 0; i < seq_len - 1; i++) {
            int p1 = seq[i], p2 = seq[i+1];
            bool found = false;
            for (int j = 0; j < num_pairs; j++) {
                if (pairs[j].pair[0] == p1 && pairs[j].pair[1] == p2) {
                    pairs[j].count++; found = true; break;
                }
            }
            if (!found) { pairs[num_pairs].pair[0] = p1; pairs[num_pairs].pair[1] = p2; pairs[num_pairs].count = 1; num_pairs++; }
        }

        // Find most frequent
        int best_idx = -1, max_count = 1;
        for (int i = 0; i < num_pairs; i++) if (pairs[i].count > max_count) { max_count = pairs[i].count; best_idx = i; }
        
        if (best_idx == -1) { free(pairs); break; }

        // Perform merge
        int p1 = pairs[best_idx].pair[0], p2 = pairs[best_idx].pair[1];
        char new_token[256];
        sprintf(new_token, "%s%s", t->vocab[p1], t->vocab[p2]);
        add_to_vocab(t, new_token);
        
        t->merges[t->num_merges++] = (MergeRule){{p1, p2}, t->vocab_size - 1};

        // Update sequence
        int *new_seq = (int *)malloc(seq_len * sizeof(int));
        int new_seq_len = 0;
        for (int i = 0; i < seq_len; i++) {
            if (i < seq_len - 1 && seq[i] == p1 && seq[i+1] == p2) {
                new_seq[new_seq_len++] = t->vocab_size - 1;
                i++;
            } else {
                new_seq[new_seq_len++] = seq[i];
            }
        }
        seq_len = new_seq_len;
        memcpy(seq, new_seq, seq_len * sizeof(int));
        free(new_seq);
        free(pairs);
    }
    free(seq);
}

int* tokenize(Tokenizer *t, const char *text, int *out_len) {
    int len = strlen(text);
    int *tokens = (int *)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) tokens[i] = (unsigned char)text[i];
    int token_len = len;
    
    // Apply merges
    for (int i = 0; i < t->num_merges; i++) {
        int *new_tokens = (int *)malloc(token_len * sizeof(int));
        int new_token_len = 0;
        for (int j = 0; j < token_len; j++) {
            if (j < token_len - 1 && tokens[j] == t->merges[i].pair[0] && tokens[j+1] == t->merges[i].pair[1]) {
                new_tokens[new_token_len++] = t->merges[i].replacement;
                j++;
            } else {
                new_tokens[new_token_len++] = tokens[j];
            }
        }
        token_len = new_token_len;
        memcpy(tokens, new_tokens, token_len * sizeof(int));
        free(new_tokens);
    }
    
    *out_len = token_len;
    return tokens;
}

char* detokenize(Tokenizer *t, int *tokens, int len) {
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
