#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct Optimizer Optimizer; // Proper forward declaration of the typedef

typedef enum {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR
} ActivationType;

typedef enum {
    MSE,
    BCE,
    CCE
} LossType;

typedef struct {
    int in_dim;
    int out_dim;
    ActivationType activation;
    Matrix weights;
    Matrix biases;
    Matrix z;           // Weighted sum before activation
    Matrix activations; // Output after activation
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
} Network;

// Lifecycle
Layer create_layer(int in_dim, int out_dim, ActivationType act);
void free_layer(Layer *l);

Network create_network(int num_layers, int *dims, ActivationType *acts);
void free_network(Network *net);

// Core Functions
void forward(Network *net, Matrix *input);
void train_step(Network *net, Matrix *input, Matrix *target, float lr, LossType loss_type, Optimizer *opt);
float activate(float x, ActivationType type);
float activate_derivative(float x, ActivationType type);

// Loss Functions
float calculate_loss(LossType type, Matrix *pred, Matrix *target);
float loss_gradient(LossType type, float pred, float target);

#endif
