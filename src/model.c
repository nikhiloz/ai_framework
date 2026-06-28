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
    model.block_inputs = (Matrix *)malloc(num_blocks * sizeof(Matrix));
    for (int i = 0; i < num_blocks; i++) {
        model.blocks[i] = create_transformer_block(num_heads, embed_dim, ffn_dim);
        model.block_inputs[i] = create_matrix(0, 0);
    }
    model.head = create_prediction_head(embed_dim, vocab_size);
    
    return model;
}

void free_model(TransformerModel *model) {
    free_tokenizer(&model->tokenizer);
    free_embedding(&model->embedding);
    for (int i = 0; i < model->num_blocks; i++) {
        free_transformer_block(&model->blocks[i]);
        free_matrix(&model->block_inputs[i]);
    }
    free(model->blocks);
    free(model->block_inputs);
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
        // Store input for backprop
        free_matrix(&model->block_inputs[i]);
        model->block_inputs[i] = copy_matrix(&current_x);
        
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
    // We need the input to prediction_backward, which is the output of the last block.
    // Since we store the input to each block, we can derive the output of the last block
    // by doing one more forward pass of that block, or better, just store it.
    // For simplicity, we'll use the last block_input and run forward once more.
    Matrix last_block_out = transformer_block_forward(&model->blocks[model->num_blocks - 1], &model->block_inputs[model->num_blocks - 1]);
    Matrix dL_dTransformerOut = prediction_backward(&model->head, &last_block_out, &logits, targets); 
    
    free_matrix(&last_block_out);
    
    // 3. Backward through Transformer Blocks (Reverse Order)
    Matrix grad = dL_dTransformerOut;
    for (int i = model->num_blocks - 1; i >= 0; i--) {
        Matrix next_grad = transformer_block_backward(&model->blocks[i], &model->block_inputs[i], &grad);
        free_matrix(&grad);
        grad = next_grad;
    }
    
    free_matrix(&grad);
    free_matrix(&logits);
}

void save_model_weights(TransformerModel *model, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for saving weights");
        return;
    }

    // Save model metadata
    fwrite(&model->vocab_size, sizeof(int), 1, fp);
    fwrite(&model->embed_dim, sizeof(int), 1, fp);
    fwrite(&model->num_blocks, sizeof(int), 1, fp);

    // Save embedding weights
    fwrite(model->embedding.weights.data, sizeof(float), model->embedding.weights.rows * model->embedding.weights.cols, fp);

    // Save transformer blocks
    for (int i = 0; i < model->num_blocks; i++) {
        TransformerBlock *tb = &model->blocks[i];
        
        // MHA Weights
        for (int h = 0; h < tb->mha.num_heads; h++) {
            fwrite(tb->mha.heads[h].W_q.data, sizeof(float), tb->mha.heads[h].W_q.rows * tb->mha.heads[h].W_q.cols, fp);
            fwrite(tb->mha.heads[h].W_k.data, sizeof(float), tb->mha.heads[h].W_k.rows * tb->mha.heads[h].W_k.cols, fp);
            fwrite(tb->mha.heads[h].W_v.data, sizeof(float), sizeof(float) * tb->mha.heads[h].W_v.rows * tb->mha.heads[h].W_v.cols, fp);
        }
        fwrite(tb->mha.W_o.data, sizeof(float), tb->mha.W_o.rows * tb->mha.W_o.cols, fp);

        // LN1
        fwrite(tb->ln1.gamma.data, sizeof(float), tb->ln1.gamma.rows * tb->ln1.gamma.cols, fp);
        fwrite(tb->ln1.beta.data, sizeof(float), tb->ln1.beta.rows * tb->ln1.beta.cols, fp);

        // FFN
        fwrite(tb->W1.data, sizeof(float), tb->W1.rows * tb->W1.cols, fp);
        fwrite(tb->b1.data, sizeof(float), tb->b1.rows * tb->b1.cols, fp);
        fwrite(tb->W2.data, sizeof(float), tb->W2.rows * tb->W2.cols, fp);
        fwrite(tb->b2.data, sizeof(float), tb->b2.rows * tb->b2.cols, fp);

        // LN2
        fwrite(tb->ln2.gamma.data, sizeof(float), tb->ln2.gamma.rows * tb->ln2.gamma.cols, fp);
        fwrite(tb->ln2.beta.data, sizeof(float), tb->ln2.beta.rows * tb->ln2.beta.cols, fp);
    }

    // Save Prediction Head
    fwrite(model->head.W_out.data, sizeof(float), model->head.W_out.rows * model->head.W_out.cols, fp);
    fwrite(model->head.b_out.data, sizeof(float), model->head.b_out.rows * model->head.b_out.cols, fp);

    fclose(fp);
}

void load_model_weights(TransformerModel *model, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file for loading weights");
        return;
    }

    int v_size, e_dim, n_blocks;
    fread(&v_size, sizeof(int), 1, fp);
    fread(&e_dim, sizeof(int), 1, fp);
    fread(&n_blocks, sizeof(int), 1, fp);

    if (v_size != model->vocab_size || e_dim != model->embed_dim || n_blocks != model->num_blocks) {
        fprintf(stderr, "Model architecture mismatch during load\n");
        fclose(fp);
        return;
    }

    // Load embedding weights
    fread(model->embedding.weights.data, sizeof(float), model->embedding.weights.rows * model->embedding.weights.cols, fp);

    // Load transformer blocks
    for (int i = 0; i < model->num_blocks; i++) {
        TransformerBlock *tb = &model->blocks[i];
        
        // MHA Weights
        for (int h = 0; h < tb->mha.num_heads; h++) {
            fread(tb->mha.heads[h].W_q.data, sizeof(float), tb->mha.heads[h].W_q.rows * tb->mha.heads[h].W_q.cols, fp);
            fread(tb->mha.heads[h].W_k.data, sizeof(float), tb->mha.heads[h].W_k.rows * tb->mha.heads[h].W_k.cols, fp);
            fread(tb->mha.heads[h].W_v.data, sizeof(float), tb->mha.heads[h].W_v.rows * tb->mha.heads[h].W_v.cols, fp);
        }
        fread(tb->mha.W_o.data, sizeof(float), tb->mha.W_o.rows * tb->mha.W_o.cols, fp);

        // LN1
        fread(tb->ln1.gamma.data, sizeof(float), tb->ln1.gamma.rows * tb->ln1.gamma.cols, fp);
        fread(tb->ln1.beta.data, sizeof(float), tb->ln1.beta.rows * tb->ln1.beta.cols, fp);

        // FFN
        fread(tb->W1.data, sizeof(float), tb->W1.rows * tb->W1.cols, fp);
        fread(tb->b1.data, sizeof(float), tb->b1.rows * tb->b1.cols, fp);
        fread(tb->W2.data, sizeof(float), tb->W2.rows * tb->W2.cols, fp);
        fread(tb->b2.data, sizeof(float), tb->b2.rows * tb->b2.cols, fp);

        // LN2
        fread(tb->ln2.gamma.data, sizeof(float), tb->ln2.gamma.rows * tb->ln2.gamma.cols, fp);
        fread(tb->ln2.beta.data, sizeof(float), tb->ln2.beta.rows * tb->ln2.beta.cols, fp);
    }

    // Load Prediction Head
    fread(model->head.W_out.data, sizeof(float), model->head.W_out.rows * model->head.W_out.cols, fp);
    fread(model->head.b_out.data, sizeof(float), model->head.b_out.rows * model->head.b_out.cols, fp);

    fclose(fp);
}
