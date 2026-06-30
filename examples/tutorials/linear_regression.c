#include "nn.h"
#include "matrix.h"
#include "optimizer.h"
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(42);
    printf("=====================================================\n");
    printf("Tutorial 1: Linear Regression (The Hello World of ML)\n");
    printf("Goal: Predict a value based on a simple linear relationship\n");
    printf("=====================================================\n\n");

    // 1. Generate a synthetic dataset: y = 2x + 1
    Dataset ds = create_dataset(100, 1, 1);
    for (int i = 0; i < 100; i++) {
        float x = (float)i / 10.0f;
        set_val(&ds.features, i, x);
        set_val(&ds.targets, i, 2.0f * x + 1.0f);
    }

    // 2. Define a Simple Linear Model
    // 1 Input -> 1 Output, Linear Activation
    int dims[] = {1, 1};
    ActivationType acts[] = {LINEAR};
    Network net = create_network(1, dims, acts);

    // 3. Setup Training
    Optimizer opt = create_optimizer(&net, ADAM, 0.01f);
    DataLoader dl = create_dataloader(&ds, 10, 1); // Batch size 10
    Matrix batch_in = create_matrix(10, 1);
    Matrix batch_tar = create_matrix(10, 1);

    printf("Training model to find the relationship y = 2x + 1...\n");

    for (int e = 0; e < 100; e++) {
        dataloader_reset(&dl);
        for (int b = 0; b < dl.num_batches; b++) {
            dataloader_next_batch(&dl, &batch_in, &batch_tar);
            train_step(&net, &batch_in, &batch_tar, opt.lr, MSE, &opt);
        }
    }

    // 4. Results & Analysis
    printf("\nTraining Complete!\n");
    printf("Learned Weight: %.4f (Target: 2.0000)\n", get_val(&net.layers[0].weights, 0));
    printf("Learned Bias:   %.4f (Target: 1.0000)\n", get_val(&net.layers[0].biases, 0));

    // Test on a new value
    Matrix test_in = create_matrix(1, 1);
    set_val(&test_in, 0, 5.0f);
    forward(&net, &test_in);
    printf("\nTest: If x = 5.0, prediction is y = %.4f (Target: 11.0000)\n", get_val(&net.layers[0].activations, 0));

    free_matrix(&batch_in);
    free_matrix(&batch_tar);
    free_matrix(&test_in);
    free_dataloader(&dl);
    free_optimizer(&opt);
    free_dataset(&ds);
    free_network(&net);
    return 0;
}
