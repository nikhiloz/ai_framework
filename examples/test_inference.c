#include "../src/tokenizer.h"
#include "../src/transformer_block.h"
#include "../src/lm_head.h"
#include "../src/sampling.h"
#include <stdio.h>

int main() {
    // 1. Setup (simplified components)
    int vocab_size = 260;
    int embed_dim = 64;
    Tokenizer t = create_tokenizer(vocab_size);
    // (Assume pre-trained or randomized weights for demo)
    TransformerBlock tb = create_transformer_block(4, embed_dim, 256);
    LMHead lm = create_lm_head(embed_dim, vocab_size);

    // 2. Input
    const char *text = "hello";
    int out_len;
    int *tokens = tokenize(&t, text, &out_len);
    
    // Convert tokens to embeddings (dummy embedding layer)
    Matrix input = create_matrix(out_len, embed_dim);
    matrix_fill_random(&input); // Placeholder for actual embedding lookup

    // 3. Inference
    Matrix tb_out = transformer_block_forward(&tb, &input);
    Matrix probs = lm_head_forward(&lm, &tb_out);

    // Sample next token
    int next_token = greedy_sample(&probs, out_len - 1);
    
    printf("Input: %s -> Next token ID: %d\n", text, next_token);

    // Cleanup
    free_matrix(&input);
    free_matrix(&tb_out);
    free_matrix(&probs);
    free(tokens);
    free_tokenizer(&t);
    free_transformer_block(&tb);
    free_lm_head(&lm);
    
    return 0;
}
