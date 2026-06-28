#ifndef CHAT_H
#define CHAT_H

typedef enum {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT
} Role;

typedef struct {
    Role role;
    char *content;
} Message;

// Formats chat history into a single model-ready prompt string
// e.g., "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nUser query [/INST]"
char* format_chat_prompt(Message *history, int history_len);

#endif
