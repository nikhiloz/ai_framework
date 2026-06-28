#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"
#include "nn.h"

typedef enum {
    SGD,
    MOMENTUM,
    ADAM
} OptimizerType;

typedef struct Optimizer {
    OptimizerType type;
    float lr;
    float beta1;     // For Momentum and Adam
    float beta2;     // For Adam
    float epsilon;   // For Adam
    int t;           // Time step for Adam
    int num_layers;  // Store number of layers for cleanup

    // State matrices for each layer
    // weight_m[i] corresponds to net->layers[i].weights
    Matrix **weight_m;
    Matrix **weight_v;
    Matrix **bias_m;
    Matrix **bias_v;
} Optimizer;

// Lifecycle
Optimizer create_optimizer(Network *net, OptimizerType type, float lr);
void free_optimizer(Optimizer *opt);

// Core Update
void optimizer_step(Optimizer *opt, Network *net, Matrix *w_grad, Matrix *b_grad, int layer_idx);

#endif
