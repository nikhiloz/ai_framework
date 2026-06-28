#include "../src/nn.h"
#include "../src/matrix.h"
#include "../src/optimizer.h"
#include "../src/data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
    srand(42);

    printf("--- Training Neural Network on XOR (Mini-Batch Pipeline) ---\n");

    // 1. Setup Dataset
    Dataset ds = create_dataset(4, 2, 1);
    float train_in[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    float train_out[4][1] = {{0}, {1}, {1}, {0}};
    for(int i=0; i<4; i++) {
        ds.features.data[i*2+0] = train_in[i][0];
        ds.features.data[i*2+1] = train_in[i][1];
        ds.targets.data[i*1+0] = train_out[i][0];
    }

    // 2. Setup Network & Optimizer
    int dims[] = {2, 8, 1};
    ActivationType acts[] = {RELU, SIGMOID};
    Network net = create_network(2, dims, acts);
    Optimizer opt = create_optimizer(&net, ADAM, 0.01f);

    // 3. Setup DataLoader
    int batch_size = 2;
    DataLoader dl = create_dataloader(&ds, batch_size, 1);

    // Pre-allocate batch matrices to avoid repeated allocation
    Matrix batch_in = create_matrix(batch_size, 2);
    Matrix batch_tar = create_matrix(batch_size, 1);

    int epochs = 5000;
    for (int e = 0; e < epochs; e++) {
        dataloader_reset(&dl);
        float epoch_loss = 0;
        int batches_processed = 0;

        for (int b = 0; b < dl.num_batches; b++) {
            dataloader_next_batch(&dl, &batch_in, &batch_tar);
            
            train_step(&net, &batch_in, &batch_tar, opt.lr, MSE, &opt);
            
            forward(&net, &batch_in);
            epoch_loss += calculate_loss(MSE, &net.layers[1].activations, &batch_tar);
            batches_processed++;
        }
        if (e % 500 == 0) {
            printf("Epoch %d, Avg MSE: %.4f\n", e, epoch_loss / batches_processed);
        }
    }

    printf("\n--- Testing Results ---\n");
    for (int i = 0; i < 4; i++) {
        Matrix input = create_matrix(1, 2);
        input.data[0] = train_in[i][0];
        input.data[1] = train_in[i][1];

        forward(&net, &input);
        printf("In: [%.0f, %.0f] -> Pred: %.4f (Target: %.0f)\n", 
               train_in[i][0], train_in[i][1], net.layers[1].activations.data[0], train_out[i][0]);
        
        free_matrix(&input);
    }

    free_matrix(&batch_in);
    free_matrix(&batch_tar);
    free_dataloader(&dl);
    free_optimizer(&opt);
    free_dataset(&ds);
    free_network(&net);
    return 0;
}
