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
    printf("Tutorial 4: Multi-Layer Perceptron (Deep Dive)\n");
    printf("Goal: Solve the non-linear XOR problem using hidden layers\n");
    printf("=====================================================\n\n");

    // 1. Dataset: XOR (The classic non-linear problem)
    Dataset ds = create_dataset(4, 2, 1);
    float train_in[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float train_out[4][1] = {{0}, {1}, {1}, {0}};
    for(int i=0; i<4; i++) {
        set_val(&ds.features, i*2+0, train_in[i][0]);
        set_val(&ds.features, i*2+1, train_in[i][1]);
        set_val(&ds.targets, i, train_out[i][0]);
    }

    // 2. Architecture: 2 In -> 8 Hidden (RELU) -> 1 Out (SIGMOID)
    int dims[] = {2, 8, 1};
    ActivationType acts[] = {RELU, SIGMOID};
    Network net = create_network(2, dims, acts);

    // 3. Training with ADAM
    Optimizer opt = create_optimizer(&net, ADAM, 0.01f);
    DataLoader dl = create_dataloader(&ds, 4, 1);
    Matrix batch_in = create_matrix(4, 2);
    Matrix batch_tar = create_matrix(4, 1);

    printf("Training MLP to solve XOR...\n");

    for (int e = 0; e < 2000; e++) {
        dataloader_reset(&dl);
        dataloader_next_batch(&dl, &batch_in, &batch_tar);
        train_step(&net, &batch_in, &batch_tar, opt.lr, MSE, &opt);
    }

    // 4. Validation
    printf("\nTesting MLP on XOR:\n");
    for (int i = 0; i < 4; i++) {
        Matrix test_in = create_matrix(1, 2);
        set_val(&test_in, 0, train_in[i][0]);
        set_val(&test_in, 1, train_in[i][1]);
        forward(&net, &test_in);
        float pred = get_val(&net.layers[1].activations, 0);
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
