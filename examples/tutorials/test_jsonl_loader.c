#include <stdio.h>
#include "chat_data.h"

int main() {
    int num_pairs = 0;
    ChatPair *pairs = load_jsonl("data/jolly_seed.jsonl", &num_pairs);
    
    if (!pairs) {
        printf("Failed to load pairs\n");
        return 1;
    }

    printf("Loaded %d pairs:\n", num_pairs);
    for (int i = 0; i < num_pairs; i++) {
        printf("Pair %d:\n", i);
        printf("  Instr: %s\n", pairs[i].instruction);
        printf("  Resp: %s\n", pairs[i].response);
    }
    
    free_chat_pairs(pairs, num_pairs);
    return 0;
}
