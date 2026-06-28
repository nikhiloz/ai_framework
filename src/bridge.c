#include "bridge.h"
#include "model.h"
#include "sampling.h"
#include <stdlib.h>
#include <string.h>

ModelHandle create_model_bridge(int vocab_size, int embed_dim, int num_heads, int num_blocks, int ffn_dim) {
    TransformerModel *model = (TransformerModel *)malloc(sizeof(TransformerModel));
    *model = create_model(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim);
    return (ModelHandle)model;
}

void free_model_bridge(ModelHandle handle) {
    TransformerModel *model = (TransformerModel *)handle;
    free_model(model);
    free(model);
}

int generate_single_token_bridge(ModelHandle handle, const char* prompt) {
    TransformerModel *model = (TransformerModel *)handle;
    
    int token_len;
    int *tokens = tokenize(&model->tokenizer, prompt, &token_len);
    
    Matrix logits = model_forward(model, tokens, token_len);
    int next_token = greedy_sample(&logits, token_len - 1);
    
    free(tokens);
    free_matrix(&logits);
    
    return next_token;
}

char* decode_token_bridge(ModelHandle handle, int token_id) {
    TransformerModel *model = (TransformerModel *)handle;
    if (token_id < 0 || token_id >= model->tokenizer.vocab_size) return "Unknown";
    return model->tokenizer.vocab[token_id];
}
