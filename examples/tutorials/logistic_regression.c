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
    printf("Tutorial 2: Logistic Regression (Binary Classification)\n");
    printf("Goal: Classify points into two categories using a sigmoid\n");
    printf("=====================================================\n\n");

    // 1. Dataset: Points (x1, x2) and Class (0 or 1)
    Dataset ds = create_dataset(100, 2, 1);
    for (int i = 0; i < 100; i++) {
        float x1 = (float)rand() / RAND_MAX;
        float x2 = (float)rand() / RAND_MAX;
        set_val(&ds.features, i*2+0, x1);
        set_val(&ds.features, i*2+1, x2);
        // Decision boundary: x1 + x2 > 1.0
        set_val(&ds.targets, i, (x1 + x2 > 1.0f) ? 1.0f : 0.0f);
    }

    // 2. Define the Model
    // 2 Inputs -> 1 Output, Sigmoid Activation
    int dims[] = {2, 1};
    ActivationType acts[] = {SIGMOID};
    Network net = create_network(1, dims, acts);

    // 3. Setup Training
    Optimizer opt = create_optimizer(&net, ADAM, 0.01f);
    DataLoader dl = create_dataloader(&ds, 10, 1);
    Matrix batch_in = create_matrix(10, 2);
    Matrix batch_tar = create_matrix(10, 1);

    printf("Training Logistic Regression model...\n");

    for (int e = 0; e < 500; e++) {
        dataloader_reset(&dl);
        for (int b = 0; b < dl.num_batches; b++) {
            dataloader_next_batch(&dl, &batch_in, &batch_tar);
            train_step(&net, &batch_in, &batch_tar, opt.lr, BCE, &opt);
        }
    }

    // 4. Testing
    printf("\nTesting Classifier:\n");
    float tests[4][2] = {{0.1, 0.1}, {0.9, 0.9}, {0.4, 0.7}, {0.2, 0.3}};
    for (int i = 0; i < 4; i++) {
        Matrix test_in = create_matrix(1, 2);
        set_val(&test_in, 0, tests[i][0]);
        set_val(&test_in, 1, tests[i][1]);
        forward(&net, &test_in);
        float pred = get_val(&net.layers[0].activations, 0);
        printf("In: [%.1f, %.1f] -> Pred: %.4f (Class: %s)\n", 
               tests[i][0], tests[i][1], pred, (pred > 0.5f) ? "1" : "0");
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
