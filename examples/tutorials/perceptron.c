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
    printf("Tutorial 3: The Perceptron (The Root of AI)\n");
    printf("Goal: Learn a simple AND logic gate\n");
    printf("=====================================================\n\n");

    // 1. Dataset: AND Gate
    Dataset ds = create_dataset(4, 2, 1);
    float train_in[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float train_out[4][1] = {{0}, {0}, {0}, {1}};
    for(int i=0; i<4; i++) {
        set_val(&ds.features, i*2+0, train_in[i][0]);
        set_val(&ds.features, i*2+1, train_in[i][1]);
        set_val(&ds.targets, i, train_out[i][0]);
    }

    // 2. Define the Model
    // A Perceptron is essentially a 1-layer network with a step-like activation
    // We use SIGMOID as a smooth approximation of the step function
    int dims[] = {2, 1};
    ActivationType acts[] = {SIGMOID};
    Network net = create_network(1, dims, acts);

    // 3. Setup Training
    Optimizer opt = create_optimizer(&net, SGD, 0.1f);
    DataLoader dl = create_dataloader(&ds, 4, 1); 
    Matrix batch_in = create_matrix(4, 2);
    Matrix batch_tar = create_matrix(4, 1);

    printf("Training Perceptron on AND gate...\n");

    for (int e = 0; e < 1000; e++) {
        dataloader_reset(&dl);
        dataloader_next_batch(&dl, &batch_in, &batch_tar);
        train_step(&net, &batch_in, &batch_tar, opt.lr, BCE, &opt);
    }

    // 4. Results
    printf("\nTesting AND Gate:\n");
    for (int i = 0; i < 4; i++) {
        Matrix test_in = create_matrix(1, 2);
        set_val(&test_in, 0, train_in[i][0]);
        set_val(&test_in, 1, train_in[i][1]);
        forward(&net, &test_in);
        float pred = get_val(&net.layers[0].activations, 0);
        printf("In: [%.0f, %.0f] -> Pred: %.4f (Class: %s)\n", 
               train_in[i][0], train_in[i][1], pred, (pred > 0.5f) ? "1" : "0");
        free_matrix(&test_in);
    }

    free_matrix(&batch_in);
    free_matrix(&batch_tar);
    free_dataloader(&dl);
    free_optimizer(&opt);
    free_dataset(&ds);
    free_network(&net);
    return 0;
}
