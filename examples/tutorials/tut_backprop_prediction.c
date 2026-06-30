#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "transformer_block.h"
#include "prediction.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("--- Prediction Head Backpropagation Test ---\n");
    
    int num_heads = 2;
    int embed_dim = 8;
    int ffn_dim = 16;
    
    Tokenizer t = create_tokenizer(100);
    const char *text = "the cat sat";
    train_tokenizer(&t, text);
    
    int token_len;
    int *tokens = tokenize(&t, text, &token_len);
    
    int *targets = (int *)malloc((token_len - 1) * sizeof(int));
    for(int i=0; i < token_len - 1; i++) targets[i] = tokens[i+1];
    
    Embedding e = create_embedding(t.vocab_size, embed_dim);
    Matrix x = create_matrix(token_len - 1, embed_dim);
    for(int i=0; i < token_len - 1; i++) {
        int tid = tokens[i];
        for(int j=0; j < embed_dim; j++) {
            set_val(&x, i * embed_dim + j, get_val(&e.weights, tid * embed_dim + j));
        }
    }
    apply_rope(&x);
    
    TransformerBlock tb = create_transformer_block(num_heads, embed_dim, ffn_dim);
    Matrix transformer_out = transformer_block_forward(&tb, &x);
    
    PredictionHead ph = create_prediction_head(embed_dim, t.vocab_size);
    
    // --- Training Iteration ---
    
    // Forward pass
    Matrix logits = prediction_forward(&ph, &transformer_out);
    float loss_before = cross_entropy_loss(&logits, targets);
    printf("Loss before update: %.4f\n", loss_before);
    
    // Backward pass
    Matrix dL_dX = prediction_backward(&ph, &transformer_out, &logits, targets);
    
    // Forward pass again to check improvement
    Matrix logits_after = prediction_forward(&ph, &transformer_out);
    float loss_after = cross_entropy_loss(&logits_after, targets);
    printf("Loss after update:  %.4f\n", loss_after);
    
    if (loss_after < loss_before) {
        printf("SUCCESS: Loss decreased!\n");
    } else {
        printf("FAILURE: Loss did not decrease.\n");
    }
    
    free(tokens);
    free(targets);
    free_matrix(&x);
    free_matrix(&transformer_out);
    free_matrix(&logits);
    free_matrix(&logits_after);
    free_matrix(&dL_dX);
    free_embedding(&e);
    free_transformer_block(&tb);
    free_prediction_head(&ph);
    free_tokenizer(&t);
    
    return 0;
}
