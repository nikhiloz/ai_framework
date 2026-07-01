#include "bridge.h"
#include "../../llama.cpp/include/llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    struct llama_vocab *vocab;
    struct llama_sampler *sampler;
    llama_token *tokens;
    int n_tokens;
} LlamaBridgeModel;

ModelHandle load_gguf_model_bridge(const char* path) {
    llama_backend_init();
    
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(path, mparams);
    if (!model) return NULL;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; 
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return NULL; }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    LlamaBridgeModel *bridge_model = (LlamaBridgeModel *)malloc(sizeof(LlamaBridgeModel));
    bridge_model->model = model;
    bridge_model->ctx = ctx;
    bridge_model->vocab = (struct llama_vocab *)llama_model_get_vocab(model);
    bridge_model->sampler = sampler;
    bridge_model->tokens = (llama_token *)malloc(2048 * sizeof(llama_token));
    bridge_model->n_tokens = 0;

    return (ModelHandle)bridge_model;
}

void free_model_bridge(ModelHandle handle) {
    LlamaBridgeModel *bridge_model = (LlamaBridgeModel *)handle;
    llama_sampler_free(bridge_model->sampler);
    llama_free(bridge_model->ctx);
    llama_model_free(bridge_model->model);
    llama_backend_free();
    free(bridge_model->tokens);
    free(bridge_model);
}

void init_generation_bridge(ModelHandle handle, const char* prompt) {
    LlamaBridgeModel *bridge_model = (LlamaBridgeModel *)handle;
    bridge_model->n_tokens = llama_tokenize(bridge_model->vocab, prompt, strlen(prompt), bridge_model->tokens, 2048, true, true);
    
}

int generate_next_token_bridge(ModelHandle handle) {
    LlamaBridgeModel *bridge_model = (LlamaBridgeModel *)handle;
    
    // Only pass the last token
    llama_token last_token = bridge_model->tokens[bridge_model->n_tokens - 1];
    struct llama_batch batch = llama_batch_get_one(&last_token, 1);
    
    if (llama_decode(bridge_model->ctx, batch) != 0) {
        return -1;
    }
    
    llama_token next_token = llama_sampler_sample(bridge_model->sampler, bridge_model->ctx, -1);
    llama_sampler_accept(bridge_model->sampler, next_token);
    
    // Append token to tokens for next iteration
    bridge_model->tokens[bridge_model->n_tokens] = next_token;
    bridge_model->n_tokens++;
    
    return next_token;
}

char* decode_token_bridge(ModelHandle handle, int token_id) {
    LlamaBridgeModel *bridge_model = (LlamaBridgeModel *)handle;
    static char buf[256];
    int n_vocab = llama_vocab_n_tokens(bridge_model->vocab);
    if (token_id < 0 || token_id >= n_vocab) return "Unknown";
    
    llama_token_to_piece(bridge_model->vocab, token_id, buf, sizeof(buf), 0, false);
    return buf;
}
