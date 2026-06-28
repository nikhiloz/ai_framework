#include "../src/transformer_block.h"
#include "../src/matrix.h"
#include "../src/optimizer.h"
#include <stdio.h>
#include <stdlib.h>

// Modular Dataset Generator
void generate_arithmetic_data(Matrix *features, Matrix *targets, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        int n = rand() % 50; // Random starting point
        features->data[i * 3 + 0] = (float)n;
        features->data[i * 3 + 1] = (float)(n + 1);
        features->data[i * 3 + 2] = (float)(n + 2);
        targets->data[i] = (float)(n + 3);
    }
}

int main() {
    srand(42);
    int num_samples = 100;
    int seq_len = 3; // Arithmetic sequence length
    int embed_dim = 4; // Simplified embedding size for tutorial

    printf("--- Tutorial: Training an Empty Transformer ---\n");

    // 1. Data Setup
    Matrix features = create_matrix(num_samples, seq_len);
    Matrix targets = create_matrix(num_samples, 1);
    generate_arithmetic_data(&features, &targets, num_samples);

    // 2. Model & Optimizer Setup
    TransformerBlock tb = create_transformer_block(1, embed_dim, 16);
    // Note: In a real training scenario, you'd need a fuller optimizer
    // and a defined loss function wrapper. Here we demonstrate the loop.

    printf("Starting training loop...\n");

    // 3. Training Loop
    int epochs = 1000;
    float lr = 0.001f;

    for (int e = 0; e < epochs; e++) {
        float total_loss = 0;
        for (int i = 0; i < num_samples; i++) {
            // Mapping scalar sequence to embeddings (simplified: 1 token = 1 embed_dim)
            Matrix input = create_matrix(seq_len, embed_dim);
            for(int s = 0; s < seq_len; s++) {
                for(int d = 0; d < embed_dim; d++) {
                    input.data[s * embed_dim + d] = features.data[i * seq_len + s];
                }
            }
            
            // Forward pass
            Matrix output = transformer_block_forward(&tb, &input);
            
            // Simple Loss: MSE against target
            float pred = output.data[0]; // Assuming output is (1, embed_dim) -> use first scalar
            float target = targets.data[i];
            float loss = 0.5f * (pred - target) * (pred - target);
            total_loss += loss;

            // Backward pass
            Matrix grad_input = transformer_block_backward(&tb, &input, &output);
            
            // Simplified: Dummy update showing the optimizer hook
            // (Assuming optimizer exists for the transformer parameters)
            
            free_matrix(&input);
            free_matrix(&output);
            free_matrix(&grad_input);
        }
        if (e % 100 == 0) printf("Epoch %d, Loss: %.4f\n", e, total_loss / num_samples);
    }

    printf("Training simulation complete.\n");

    // Cleanup
    free_matrix(&features);
    free_matrix(&targets);
    free_transformer_block(&tb);
    return 0;
}
