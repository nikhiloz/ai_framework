#include "model.h"
#include "chat_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void train_on_pair(TransformerModel *model, const char *instr, const char *resp, float lr) {
    printf("Debug: Training on instr: %s\n", instr);
    char combined[4096];
    snprintf(combined, sizeof(combined), "Instruction: %s\nResponse: %s", instr, resp);
    
    printf("Debug: Tokenizing...\n");
    int token_len = 0;
    int *tokens = tokenize(&model->tokenizer, combined, &token_len);
    printf("Debug: Tokenized to len: %d\n", token_len);
    
    if (tokens == NULL || token_len < 2) {
        printf("Debug: Skipping (too short or NULL)\n");
        if (tokens) free(tokens);
        return;
    }

    int *targets = (int *)malloc((token_len - 1) * sizeof(int));
    if (!targets) {
        printf("Debug: Malloc failed\n");
        free(tokens);
        return;
    }
    for (int i = 0; i < token_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }

    printf("Debug: Stepping...\n");
    model_train_step(model, tokens, targets, token_len - 1, lr);
    printf("Debug: Step done.\n");

    free(tokens);
    free(targets);
}

int main() {
    int vocab_size = 256;
    int embed_dim = 32;
    int num_heads = 4;
    int num_blocks = 2;
    int ffn_dim = 64;
    float lr = 0.01f;
    int epochs = 100;

    printf("--- Jolly-Bot Training Harness ---\n");
    
    int num_pairs = 0;
    printf("Attempting to load data from: ai_framework/data/jolly_seed.jsonl\n");
    ChatPair *pairs = load_jsonl("/data/data/com.termux/files/home/ai_framework/data/jolly_seed.jsonl", &num_pairs);
    if (!pairs) {
        printf("Failed to load training data!\n");
        return 1;
    }
    
    TransformerModel model = create_model(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim);
    
    printf("Initializing tokenizer...\n");
    for (int i = 0; i < num_pairs; i++) {
        char combined[4096];
        snprintf(combined, sizeof(combined), "Instruction: %s\nResponse: %s", pairs[i].instruction, pairs[i].response);
        train_tokenizer(&model.tokenizer, combined);
    }
    
    printf("Training on %d pairs for %d epochs...\n", num_pairs, epochs);
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_pairs; i++) {
            train_on_pair(&model, pairs[i].instruction, pairs[i].response, lr);
        }
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("Epoch %d/%d completed\n", epoch + 1, epochs);
        }
    }
    
    printf("\n--- Testing Generation ---\n");
    const char *prompt = "Instruction: Hello!\nResponse: ";
    int generated = model_generate(&model, prompt, 20);
    printf("Prompt: %s\n", prompt);
    printf("Generated token ID: %d\n", generated);
    
    const char *weights_file = "jolly_weights.bin";
    printf("Saving weights to %s...\n", weights_file);
    save_model_weights(&model, weights_file);
    
    free_chat_pairs(pairs, num_pairs);
    free_model(&model);
    
    printf("\nTraining complete. Weights saved.\n");
    return 0;
}
