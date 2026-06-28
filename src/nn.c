#include "nn.h"
#include "matrix.h"
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float activate(float x, ActivationType type) {
    switch (type) {
        case SIGMOID: return 1.0f / (1.0f + expf(-x));
        case RELU:    return (x > 0) ? x : 0;
        case TANH:    return tanhf(x);
        case SOFTMAX: return x; // Handled in forward() for layer-wide normalization
        case LINEAR:  return x;
        default:      return x;
    }
}

float activate_derivative(float x, ActivationType type) {
    switch (type) {
        case SIGMOID: {
            float s = activate(x, SIGMOID);
            return s * (1.0f - s);
        }
        case RELU: {
            return (x > 0) ? 1.0f : 0.0f;
        }
        case TANH: {
            float t = tanhf(x);
            return 1.0f - (t * t);
        }
        case LINEAR: return 1.0f;
        default: return 1.0f;
    }
}

Layer create_layer(int in_dim, int out_dim, ActivationType act) {
    Layer l;
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    l.activation = act;
    l.weights = create_matrix(in_dim, out_dim);
    matrix_fill_random(&l.weights);
    l.biases = create_matrix(1, out_dim);
    matrix_fill_zero(&l.biases);
    l.z = create_matrix(1, out_dim);
    l.activations = create_matrix(1, out_dim);
    return l;
}

void free_layer(Layer *l) {
    free_matrix(&l->weights);
    free_matrix(&l->biases);
    free_matrix(&l->z);
    free_matrix(&l->activations);
}

Network create_network(int num_layers, int *dims, ActivationType *acts) {
    Network net;
    net.num_layers = num_layers;
    net.layers = (Layer *)malloc(num_layers * sizeof(Layer));
    for (int i = 0; i < num_layers; i++) {
        net.layers[i] = create_layer(dims[i], dims[i+1], acts[i]);
    }
    return net;
}

void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
}

void forward(Network *net, Matrix *input) {
    Matrix *current_input = input;
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        
        // Re-allocate z and activations if batch size changes
        if (l->z.rows != input->rows) {
            free_matrix(&l->z);
            free_matrix(&l->activations);
            l->z = create_matrix(input->rows, l->out_dim);
            l->activations = create_matrix(input->rows, l->out_dim);
        }
        
        Matrix z_prod = matrix_multiply(current_input, &l->weights);
        matrix_add_bias(&z_prod, &l->biases, &l->z);
        
        if (l->activation == SOFTMAX) {
            // Softmax needs to be computed per-row for mini-batches
            for (int row = 0; row < l->z.rows; row++) {
                float max_z = l->z.data[row * l->z.cols];
                for (int j = 1; j < l->z.cols; j++) {
                    if (l->z.data[row * l->z.cols + j] > max_z) max_z = l->z.data[row * l->z.cols + j];
                }
                
                float sum = 0;
                for (int j = 0; j < l->z.cols; j++) {
                    l->activations.data[row * l->z.cols + j] = expf(l->z.data[row * l->z.cols + j] - max_z);
                    sum += l->activations.data[row * l->z.cols + j];
                }
                for (int j = 0; j < l->z.cols; j++) {
                    l->activations.data[row * l->z.cols + j] /= sum;
                }
            }
        } else {
            for (int j = 0; j < l->z.rows * l->z.cols; j++) {
                l->activations.data[j] = activate(l->z.data[j], l->activation);
            }
        }
        free_matrix(&z_prod);
        current_input = &l->activations;
    }
}

// Backpropagation Implementation
void train_step(Network *net, Matrix *input, Matrix *target, float lr, LossType loss_type, Optimizer *opt) {
    int batch_size = input->rows;
    opt->lr = lr;
    
    // 1. Forward pass for the whole batch
    forward(net, input);

    // 2. Compute Output Layer Error (Delta) for the batch
    Layer *out_layer = &net->layers[net->num_layers - 1];
    Matrix delta = create_matrix(batch_size, out_layer->out_dim);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_layer->out_dim; j++) {
            float pred = out_layer->activations.data[i * out_layer->out_dim + j];
            float gradient = loss_gradient(loss_type, pred, target->data[i * out_layer->out_dim + j]);
            delta.data[i * out_layer->out_dim + j] = gradient * activate_derivative(out_layer->z.data[i * out_layer->out_dim + j], out_layer->activation);
        }
    }

    // 3. Propagate error backwards
    for (int i = net->num_layers - 1; i >= 0; i--) {
        Layer *l = &net->layers[i];
        Matrix *prev_activations = (i == 0) ? input : &net->layers[i-1].activations;
        
        // Calculate Gradient for Weights: dL/dW = Input^T * Delta
        // Result is (in_dim x batch_size) * (batch_size x out_dim) = (in_dim x out_dim)
        Matrix prev_trans = matrix_transpose(prev_activations);
        Matrix w_grad = matrix_multiply(&prev_trans, &delta);
        
        // Average weight gradients over the batch
        matrix_scalar_multiply(&w_grad, 1.0f / batch_size);
        
        // Calculate Gradient for Bias: dL/dB = sum(Delta) over batch
        Matrix b_grad = create_matrix(1, l->biases.cols);
        matrix_fill_zero(&b_grad);
        for (int b = 0; b < batch_size; b++) {
            for (int j = 0; j < l->biases.cols; j++) {
                b_grad.data[j] += delta.data[b * l->biases.cols + j];
            }
        }
        // Average bias gradients over the batch
        matrix_scalar_multiply(&b_grad, 1.0f / batch_size);

        // Use the Optimizer to update weights and biases
        optimizer_step(opt, net, &w_grad, &b_grad, i);

        // Calculate Delta for the previous layer
        if (i > 0) {
            Matrix w_trans = matrix_transpose(&l->weights);
            // (batch_size x out_dim) * (out_dim x in_dim) = (batch_size x in_dim)
            Matrix prev_delta_raw = matrix_multiply(&delta, &w_trans);
            
            Layer *prev = &net->layers[i-1];
            Matrix next_delta = create_matrix(batch_size, prev->out_dim);
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < prev->out_dim; j++) {
                    float z_val = prev->z.data[b * prev->out_dim + j];
                    next_delta.data[b * prev->out_dim + j] = prev_delta_raw.data[b * prev->out_dim + j] * activate_derivative(z_val, prev->activation);
                }
            }
            
            free_matrix(&prev_delta_raw);
            free_matrix(&w_trans);
            
            free_matrix(&delta);
            delta = next_delta;
        } else {
            free_matrix(&delta);
        }
        
        free_matrix(&prev_trans);
        free_matrix(&w_grad);
        free_matrix(&b_grad);
    }
}

float calculate_loss(LossType type, Matrix *pred, Matrix *target) {
    float loss = 0;
    int n = pred->cols;
    for (int i = 0; i < n; i++) {
        float p = pred->data[i];
        float t = target->data[i];
        switch (type) {
            case MSE: loss += 0.5f * (p - t) * (p - t); break;
            case BCE: {
                p = (p < 1e-7f) ? 1e-7f : (p > 1.0f - 1e-7f) ? 1.0f - 1e-7f : p;
                loss -= (t * logf(p) + (1.0f - t) * logf(1.0f - p));
                break;
            }
            case CCE: {
                p = (p < 1e-7f) ? 1e-7f : p;
                loss -= t * logf(p);
                break;
            }
        }
    }
    return loss / n;
}

float loss_gradient(LossType type, float pred, float target) {
    switch (type) {
        case MSE: return pred - target;
        case BCE: {
            float p = (pred < 1e-7f) ? 1e-7f : (pred > 1.0f - 1e-7f) ? 1.0f - 1e-7f : pred;
            return (p - target) / (p * (1.0f - p));
        }
        case CCE: {
            float p = (pred < 1e-7f) ? 1e-7f : pred;
            return -target / p;
        }
    }
    return 0;
}
