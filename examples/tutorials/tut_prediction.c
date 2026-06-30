#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "transformer_block.h"
#include "prediction.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("--- Next Token Prediction Test ---\n");
    
    // Hyperparameters
    int num_heads = 2;
    int embed_dim = 8;
    int ffn_dim = 16;
    
    // 1. Text and Tokenization
    const char *text = "the cat sat on the mat";
    Tokenizer t = create_tokenizer(100);
    train_tokenizer(&t, text);
    
    int token_len;
    int *tokens = tokenize(&t, text, &token_len);
    
    // Targets for "Next Token Prediction": shift tokens by 1
    // Input: [t0, t1, t2] -> Target: [t1, t2, t3]
    int *targets = (int *)malloc((token_len - 1) * sizeof(int));
    for(int i=0; i < token_len - 1; i++) {
        targets[i] = tokens[i+1];
    }
    
    // 2. Foundation Pipeline
    Embedding e = create_embedding(t.vocab_size, embed_dim);
    Matrix x = create_matrix(token_len - 1, embed_dim);
    // Only use tokens up to token_len - 1 for input
    for(int i=0; i < token_len - 1; i++) {
        // We need to look up the token from the original tokens array
        // Simplified: manually copying into a matrix for this test
        int tid = tokens[i];
        for(int j=0; j < embed_dim; j++) {
            set_val(&x, i * embed_dim + j, get_val(&e.weights, tid * embed_dim + j));
        }
    }
    apply_rope(&x);
    
    // 3. Transformer Block
    TransformerBlock tb = create_transformer_block(num_heads, embed_dim, ffn_dim);
    Matrix transformer_out = transformer_block_forward(&tb, &x);
    
    // 4. Prediction Head
    PredictionHead ph = create_prediction_head(embed_dim, t.vocab_size);
    Matrix logits = prediction_forward(&ph, &transformer_out);
    
    // 5. Calculate Loss
    float loss = cross_entropy_loss(&logits, targets);
    
    printf("Sequence: %s\n", text);
    printf("Input length: %d, Target length: %d\n", token_len - 1, token_len - 1);
    printf("Initial Cross-Entropy Loss: %.4f\n", loss);
    
    // Cleanup
    free(tokens);
    free(targets);
    free_matrix(&x);
    free_matrix(&transformer_out);
    free_matrix(&logits);
    free_embedding(&e);
    free_transformer_block(&tb);
    free_prediction_head(&ph);
    free_tokenizer(&t);
    
    return 0;
}
