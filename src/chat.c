#include "chat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* format_chat_prompt(Message *history, int history_len) {
    // Rough estimate of size to avoid repeated reallocs
    size_t total_size = 1024; 
    char *prompt = (char *)malloc(total_size);
    prompt[0] = '\0';

    for (int i = 0; i < history_len; i++) {
        char *role_str;
        switch (history[i].role) {
            case ROLE_SYSTEM: role_str = "SYSTEM"; break;
            case ROLE_USER: role_str = "USER"; break;
            case ROLE_ASSISTANT: role_str = "ASSISTANT"; break;
        }
        
        char formatted_msg[512];
        snprintf(formatted_msg, sizeof(formatted_msg), "[%s]: %s\n", role_str, history[i].content);
        strcat(prompt, formatted_msg);
    }
    strcat(prompt, "[ASSISTANT]: ");
    
    return prompt;
}
