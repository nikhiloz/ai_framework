#include "chat_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 2048

// Minimal JSON parser for our specific "instruction"/"response" format
ChatPair *load_jsonl(const char *filename, int *num_pairs) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    int capacity = 10;
    int count = 0;
    ChatPair *pairs = malloc(capacity * sizeof(ChatPair));
    char line[MAX_LINE_LENGTH];

    while (fgets(line, sizeof(line), file)) {
        if (count >= capacity) {
            capacity *= 2;
            pairs = realloc(pairs, capacity * sizeof(ChatPair));
        }

        // Extremely naive parsing for {"instruction": "...", "response": "..."}
        char *instr_start = strstr(line, "\"instruction\": \"");
        char *resp_start = strstr(line, "\"response\": \"");
        
        if (instr_start && resp_start) {
            instr_start += 16;
            char *instr_end = strchr(instr_start, '\"');
            
            resp_start += 13;
            char *resp_end = strchr(resp_start, '\"');

            if (instr_end && resp_end) {
                *instr_end = '\0';
                *resp_end = '\0';
                pairs[count].instruction = strdup(instr_start);
                pairs[count].response = strdup(resp_start);
                count++;
            }
        }
    }

    fclose(file);
    *num_pairs = count;
    return pairs;
}

void free_chat_pairs(ChatPair *pairs, int num_pairs) {
    for (int i = 0; i < num_pairs; i++) {
        free(pairs[i].instruction);
        free(pairs[i].response);
    }
    free(pairs);
}
