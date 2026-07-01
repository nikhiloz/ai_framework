#include "bridge.h"
#include "model.h"
#include "sampling.h"
#include <stdlib.h>
#include <string.h>

// ... existing imports ...
#include "gguf_loader.h"

ModelHandle create_model_bridge(int vocab_size, int embed_dim, int num_heads, int num_blocks, int ffn_dim) {
    TransformerModel *model = (TransformerModel *)malloc(sizeof(TransformerModel));
    *model = create_model(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim);
    return (ModelHandle)model;
}

ModelHandle load_gguf_model_bridge(const char* path) {
    printf("DEBUG: load_gguf_model_bridge: calling load_model_gguf\n");
    fflush(stdout);
    TransformerModel *model = malloc(sizeof(TransformerModel));
    *model = load_model_gguf(path);
    printf("DEBUG: load_gguf_model_bridge: load_model_gguf returned\n");
    fflush(stdout);
    return (ModelHandle)model;
}

void free_model_bridge(ModelHandle handle) {
// ... (rest of file)
    TransformerModel *model = (TransformerModel *)handle;
    free_model(model);
    free(model);
}

int generate_single_token_bridge(ModelHandle handle, const char* prompt) {
    TransformerModel *model = (TransformerModel *)handle;
    printf("DEBUG: generate_single_token_bridge started. Prompt: %s, model=%p\n", prompt, (void*)model);
    
    int token_len;
    int *tokens = tokenize(&model->tokenizer, prompt, &token_len);
    printf("DEBUG: Tokenized prompt. Length: %d\n", token_len);
    
    printf("DEBUG: Calling model_forward...\n");
    Matrix logits = model_forward(model, tokens, token_len);
    printf("DEBUG: model_forward returned. Logits rows: %d, cols: %d\n", logits.rows, logits.cols);
    
    printf("DEBUG: Calling greedy_sample...\n");
    int next_token = greedy_sample(&logits, token_len - 1);
    printf("DEBUG: greedy_sample returned token: %d\n", next_token);
    
    free(tokens);
    free_matrix(&logits);
    
    return next_token;
}

char* decode_token_bridge(ModelHandle handle, int token_id) {
    TransformerModel *model = (TransformerModel *)handle;
    if (token_id < 0 || token_id >= model->tokenizer.vocab_size) return "Unknown";
    return model->tokenizer.vocab[token_id];
}
