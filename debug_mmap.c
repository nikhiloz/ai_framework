#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "src/model.h"
#include "src/gguf_loader.h"
#include "src/matrix.h"

int main() {
    const char* filename = "tinyllama.gguf";
    printf("Starting GGUF load test...\n");
    
    TransformerModel model = load_model_gguf(filename);
    printf("Model loaded successfully. Vocab: %d, Dim: %d\n", model.vocab_size, model.embed_dim);
    
    printf("Testing embedding access...\n");
    float val = get_val(&model.embedding.weights, 0);
    printf("First embedding weight: %f\n", val);
    
    printf("Testing block 0 weight access...\n");
    float b_val = get_val(&model.blocks[0].W1, 0);
    printf("First block 0 W1 weight: %f\n", b_val);
    
    printf("Testing prediction head access...\n");
    float h_val = get_val(&model.head.W_out, 0);
    printf("First head weight: %f\n", h_val);
    
    printf("All basic accesses successful. Freeing model...\n");
    free_model(&model);
    printf("Done.\n");
    
    return 0;
}
