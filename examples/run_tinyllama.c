#include "../src/gguf_loader.h"
#include "../src/model.h"
#include "../src/matrix.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    printf("--- VERIFYING BINARY: BUILD_ID_X99 ---\n");
    if (argc < 2) {
        printf("Usage: %s <tinyllama.gguf>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    printf("Attempting to load model: %s\n", model_path);

    // Load model using GGUF loader
    TransformerModel model = load_model_gguf(model_path);
    
    printf("Model loaded successfully.\n");
    printf("Vocab Size: %d, Embed Dim: %d, Num Blocks: %d\n", 
            model.vocab_size, model.embed_dim, model.num_blocks);

    // Inference Test:
    // Since the tokenizer is not yet loaded with TinyLlama's BPE vocab,
    // we'll use a dummy sequence of tokens to verify the forward pass.
    int test_tokens[] = {1, 10, 100, 1000};
    int seq_len = 4;
    printf("Running forward pass with dummy tokens: [1, 10, 100, 1000]...\n");

    Matrix logits = model_forward(&model, test_tokens, seq_len);
    
    printf("Forward pass complete. Logits shape: (%d, %d)\n", logits.rows, logits.cols);
    
    // Print a few values from the logits of the last token to verify they are numbers
    printf("Last token logits (first 5 values): ");
    float *logits_data = (float *)logits.data;
    for (int i = 0; i < 5; i++) {
        printf("%f ", logits_data[(seq_len - 1) * logits.cols + i]);
    }
    printf("\n");

    free_matrix(&logits);
    printf("Cleaning up model...\n");
    free_model(&model);
    printf("Cleanup complete.\n");

    return 0;
}
