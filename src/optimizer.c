#include "optimizer.h"
#include <stdlib.h>
#include <math.h>

Optimizer create_optimizer(Network *net, OptimizerType type, float lr) {
    Optimizer opt;
    opt.type = type;
    opt.lr = lr;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.epsilon = 1e-8f;
    opt.t = 0;
    opt.num_layers = net->num_layers;

    opt.weight_m = (Matrix **)malloc(opt.num_layers * sizeof(Matrix *));
    opt.weight_v = (Matrix **)malloc(opt.num_layers * sizeof(Matrix *));
    opt.bias_m = (Matrix **)malloc(opt.num_layers * sizeof(Matrix *));
    opt.bias_v = (Matrix **)malloc(opt.num_layers * sizeof(Matrix *));

    for (int i = 0; i < opt.num_layers; i++) {
        Layer *l = &net->layers[i];
        
        opt.weight_m[i] = NULL;
        opt.weight_v[i] = NULL;
        opt.bias_m[i] = NULL;
        opt.bias_v[i] = NULL;

        if (type == MOMENTUM || type == ADAM) {
            opt.weight_m[i] = malloc(sizeof(Matrix));
            *opt.weight_m[i] = create_matrix(l->weights.rows, l->weights.cols);
            matrix_fill_zero(opt.weight_m[i]);

            opt.bias_m[i] = malloc(sizeof(Matrix));
            *opt.bias_m[i] = create_matrix(l->biases.rows, l->biases.cols);
            matrix_fill_zero(opt.bias_m[i]);
        }

        if (type == ADAM) {
            opt.weight_v[i] = malloc(sizeof(Matrix));
            *opt.weight_v[i] = create_matrix(l->weights.rows, l->weights.cols);
            matrix_fill_zero(opt.weight_v[i]);

            opt.bias_v[i] = malloc(sizeof(Matrix));
            *opt.bias_v[i] = create_matrix(l->biases.rows, l->biases.cols);
            matrix_fill_zero(opt.bias_v[i]);
        }
    }

    return opt;
}

void free_optimizer(Optimizer *opt) {
    for (int i = 0; i < opt->num_layers; i++) {
        if (opt->weight_m[i]) {
            free_matrix(opt->weight_m[i]);
            free(opt->weight_m[i]);
        }
        if (opt->weight_v[i]) {
            free_matrix(opt->weight_v[i]);
            free(opt->weight_v[i]);
        }
        if (opt->bias_m[i]) {
            free_matrix(opt->bias_m[i]);
            free(opt->bias_m[i]);
        }
        if (opt->bias_v[i]) {
            free_matrix(opt->bias_v[i]);
            free(opt->bias_v[i]);
        }
    }
    free(opt->weight_m);
    free(opt->weight_v);
    free(opt->bias_m);
    free(opt->bias_v);
}

void optimizer_step(Optimizer *opt, Network *net, Matrix *w_grad, Matrix *b_grad, int layer_idx) {
    Layer *l = &net->layers[layer_idx];
    
    if (opt->type == SGD) {
        for (int i = 0; i < l->weights.rows * l->weights.cols; i++) {
            l->weights.data[i] -= opt->lr * w_grad->data[i];
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            l->biases.data[i] -= opt->lr * b_grad->data[i];
        }
    } else if (opt->type == MOMENTUM) {
        Matrix *wm = opt->weight_m[layer_idx];
        Matrix *bm = opt->bias_m[layer_idx];
        
        for (int i = 0; i < l->weights.rows * l->weights.cols; i++) {
            wm->data[i] = opt->beta1 * wm->data[i] + (1.0f - opt->beta1) * w_grad->data[i];
            l->weights.data[i] -= opt->lr * wm->data[i];
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            bm->data[i] = opt->beta1 * bm->data[i] + (1.0f - opt->beta1) * b_grad->data[i];
            l->biases.data[i] -= opt->lr * bm->data[i];
        }
    } else if (opt->type == ADAM) {
        opt->t++;
        Matrix *wm = opt->weight_m[layer_idx];
        Matrix *wv = opt->weight_v[layer_idx];
        Matrix *bm = opt->bias_m[layer_idx];
        Matrix *bv = opt->bias_v[layer_idx];

        float bias_corr1 = 1.0f - powf(opt->beta1, opt->t);
        float bias_corr2 = 1.0f - powf(opt->beta2, opt->t);

        for (int i = 0; i < l->weights.rows * l->weights.cols; i++) {
            wm->data[i] = opt->beta1 * wm->data[i] + (1.0f - opt->beta1) * w_grad->data[i];
            wv->data[i] = opt->beta2 * wv->data[i] + (1.0f - opt->beta2) * w_grad->data[i] * w_grad->data[i];
            
            float m_hat = wm->data[i] / bias_corr1;
            float v_hat = wv->data[i] / bias_corr2;
            l->weights.data[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->epsilon);
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            bm->data[i] = opt->beta1 * bm->data[i] + (1.0f - opt->beta1) * b_grad->data[i];
            bv->data[i] = opt->beta2 * bv->data[i] + (1.0f - opt->beta2) * b_grad->data[i] * b_grad->data[i];
            
            float m_hat = bm->data[i] / bias_corr1;
            float v_hat = bv->data[i] / bias_corr2;
            l->biases.data[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->epsilon);
        }
    }
}
