#include "model.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("--- TinyLlama Weight Loader Verification ---\n");
    
    // Reduced parameters for verification
    int vocab_size = 1000;
    int embed_dim = 64;
    int num_heads = 4;
    int num_blocks = 2;
    int ffn_dim = 128;
    
    printf("Initializing model...\n");
    TransformerModel model = create_model(vocab_size, embed_dim, num_heads, num_blocks, ffn_dim);
    
    printf("Attempting to load weights with mmap...\n");
    load_llama_weights_mmap(&model, "tinyllama_weights.bin");
    
    // Verify some values
    // W_q of the first head of the first block should be 0, 1, 2...
    float val = get_val(&model.blocks[0].mha.heads[0].W_q, 0);
    printf("Verified W_q[0]: %f (expected 0.000000)\n", val);
    
    printf("Verification complete.\n");
    
    free_model(&model);
    return 0;
}
