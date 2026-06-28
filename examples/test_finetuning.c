#include "../src/chat.h"
#include "../src/tokenizer.h"
#include "../src/transformer_block.h"
#include "../src/lm_head.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char *instruction;
    char *response;
} InstructionPair;

int main() {
    // 1. Setup
    int vocab_size = 260;
    int embed_dim = 64;
    Tokenizer t = create_tokenizer(vocab_size);
    TransformerBlock tb = create_transformer_block(4, embed_dim, 256);
    LMHead lm = create_lm_head(embed_dim, vocab_size);

    // 2. Sample Data
    InstructionPair data = {"Calculate 1+1", "2"};
    Message history[2] = {{ROLE_USER, data.instruction}, {ROLE_ASSISTANT, data.response}};
    
    // 3. Prompt Formatting
    char *prompt = format_chat_prompt(history, 2);
    printf("Formatted Prompt:\n%s\n", prompt);

    // 4. Tokenization & Simulation
    int out_len;
    int *tokens = tokenize(&t, prompt, &out_len);
    printf("Tokenized length: %d\n", out_len);

    // 5. Training Step Simulation
    // In actual implementation, we would:
    // a. Forward pass over full prompt
    // b. Mask loss for [USER] part
    // c. Calculate loss for [ASSISTANT] part
    // d. Backward pass
    printf("Simulating Fine-Tuning backward pass on Assistant response...\n");

    // Cleanup
    free(prompt);
    free(tokens);
    free_tokenizer(&t);
    free_transformer_block(&tb);
    free_lm_head(&lm);
    
    return 0;
}
