#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sampling.h"

TransformerModel create_model(int vocab_size, int embed_dim, int num_heads, int num_blocks, int ffn_dim) {
    TransformerModel model;
    model.vocab_size = vocab_size;
    model.embed_dim = embed_dim;
    model.tokenizer = create_tokenizer(vocab_size * 2); 
    model.embedding = create_embedding(vocab_size, embed_dim);
    model.num_blocks = num_blocks;
    model.blocks = (TransformerBlock *)malloc(num_blocks * sizeof(TransformerBlock));
    for (int i = 0; i < num_blocks; i++) {
        model.blocks[i] = create_transformer_block(num_heads, embed_dim, ffn_dim);
    }
    model.head = create_prediction_head(embed_dim, vocab_size);
    
    return model;
}

void free_model(TransformerModel *model) {
    free_tokenizer(&model->tokenizer);
    free_embedding(&model->embedding);
    for (int i = 0; i < model->num_blocks; i++) {
        free_transformer_block(&model->blocks[i]);
    }
    free(model->blocks);
    free_prediction_head(&model->head);
}

Matrix model_forward(TransformerModel *model, int *tokens, int seq_len) {
    // 1. Embedding
    Matrix x = create_matrix(seq_len, model->embed_dim);
    embedding_lookup(&model->embedding, tokens, seq_len, &x);
    
    // 2. Positional Encoding
    apply_positional_encoding(&x);
    
    // 3. Transformer Blocks
    Matrix current_x = x;
    for (int i = 0; i < model->num_blocks; i++) {
        Matrix next_x = transformer_block_forward(&model->blocks[i], &current_x);
        if (i > 0) free_matrix(&current_x);
        current_x = next_x;
    }
    
    // 4. Prediction Head
    Matrix logits = prediction_forward(&model->head, &current_x);
    
    free_matrix(&x);
    free_matrix(&current_x);
    
    return logits;
}

int model_generate(TransformerModel *model, const char *prompt, int max_len) {
    int token_len;
    int *tokens = tokenize(&model->tokenizer, prompt, &token_len);
    
    for (int i = 0; i < max_len; i++) {
        Matrix logits = model_forward(model, tokens, token_len);
        
        // Sample next token (Greedy)
        int next_token = greedy_sample(&logits, token_len - 1);
        
        tokens = realloc(tokens, (token_len + 1) * sizeof(int));
        tokens[token_len] = next_token;
        token_len++;
        
        free_matrix(&logits);
        if (next_token == 0) break; // Assume 0 is EOS
    }
    
    int final_token = tokens[token_len - 1];
    free(tokens);
    return final_token;
}

void model_train_step(TransformerModel *model, int *tokens, int *targets, int seq_len, float lr) {
    // 1. Forward
    Matrix logits = model_forward(model, tokens, seq_len);
    
    // 2. Backward through Prediction Head
    Matrix dL_dTransformerOut = prediction_backward(&model->head, NULL, &logits, targets); 
    // Note: We need the input to prediction_backward, which is the output of the last block.
    // This is a simplification for this pedagogical framework.
    
    // 3. Backward through Transformer Blocks (Reverse Order)
    Matrix grad = dL_dTransformerOut;
    for (int i = model->num_blocks - 1; i >= 0; i--) {
        // We need the original input to each block for backward
        // For now, we just pass a dummy as the framework focuses on block-level backward
        Matrix next_grad = transformer_block_backward(&model->blocks[i], NULL, &grad);
        free_matrix(&grad);
        grad = next_grad;
    }
    
    free_matrix(&grad);
    free_matrix(&logits);
}
