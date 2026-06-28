#include "../src/tokenizer.h"
#include "../src/transformer_block.h"
#include "../src/lm_head.h"
#include "../src/sampling.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Setup
    int vocab_size = 260;
    int embed_dim = 64;
    int seq_len = 10;
    Tokenizer t = create_tokenizer(vocab_size);
    TransformerBlock tb = create_transformer_block(4, embed_dim, 256);
    LMHead lm = create_lm_head(embed_dim, vocab_size);

    // 2. Input
    const char *text = "hello";
    int out_len;
    int *tokens = tokenize(&t, text, &out_len);
    
    printf("Input: %s\nGenerating: ", text);

    // 3. Inference Loop
    int max_gen = 5;
    int *gen_tokens = (int *)malloc((out_len + max_gen) * sizeof(int));
    memcpy(gen_tokens, tokens, out_len * sizeof(int));
    int current_len = out_len;

    for (int step = 0; step < max_gen; step++) {
        // Embed input (placeholder)
        Matrix input = create_matrix(current_len, embed_dim);
        matrix_fill_random(&input); 

        // Forward
        Matrix tb_out = transformer_block_forward(&tb, &input);
        Matrix probs = lm_head_forward(&lm, &tb_out);

        // Sample
        int next_token = greedy_sample(&probs, current_len - 1);
        gen_tokens[current_len++] = next_token;
        printf("%d ", next_token);

        free_matrix(&input);
        free_matrix(&tb_out);
        free_matrix(&probs);
    }
    
    printf("\n");

    // Cleanup
    free(tokens);
    free(gen_tokens);
    free_tokenizer(&t);
    free_transformer_block(&tb);
    free_lm_head(&lm);
    
    return 0;
}
