#ifndef CHAT_DATA_H
#define CHAT_DATA_H

typedef struct {
    char *instruction;
    char *response;
} ChatPair;

ChatPair *load_jsonl(const char *filename, int *num_pairs);
void free_chat_pairs(ChatPair *pairs, int num_pairs);

#endif
