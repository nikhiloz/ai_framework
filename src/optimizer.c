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
            set_val(&l->weights, i, get_val(&l->weights, i) - opt->lr * get_val(w_grad, i));
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            set_val(&l->biases, i, get_val(&l->biases, i) - opt->lr * get_val(b_grad, i));
        }
    } else if (opt->type == MOMENTUM) {
        Matrix *wm = opt->weight_m[layer_idx];
        Matrix *bm = opt->bias_m[layer_idx];
        
        for (int i = 0; i < l->weights.rows * l->weights.cols; i++) {
            float new_m = opt->beta1 * get_val(wm, i) + (1.0f - opt->beta1) * get_val(w_grad, i);
            set_val(wm, i, new_m);
            set_val(&l->weights, i, get_val(&l->weights, i) - opt->lr * new_m);
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            float new_m = opt->beta1 * get_val(bm, i) + (1.0f - opt->beta1) * get_val(b_grad, i);
            set_val(bm, i, new_m);
            set_val(&l->biases, i, get_val(&l->biases, i) - opt->lr * new_m);
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
            float wg = get_val(w_grad, i);
            float new_m = opt->beta1 * get_val(wm, i) + (1.0f - opt->beta1) * wg;
            float new_v = opt->beta2 * get_val(wv, i) + (1.0f - opt->beta2) * wg * wg;
            set_val(wm, i, new_m);
            set_val(wv, i, new_v);
            
            float m_hat = new_m / bias_corr1;
            float v_hat = new_v / bias_corr2;
            set_val(&l->weights, i, get_val(&l->weights, i) - opt->lr * m_hat / (sqrtf(v_hat) + opt->epsilon));
        }
        for (int i = 0; i < l->biases.rows * l->biases.cols; i++) {
            float bg = get_val(b_grad, i);
            float new_m = opt->beta1 * get_val(bm, i) + (1.0f - opt->beta1) * bg;
            float new_v = opt->beta2 * get_val(bv, i) + (1.0f - opt->beta2) * bg * bg;
            set_val(bm, i, new_m);
            set_val(bv, i, new_v);
            
            float m_hat = new_m / bias_corr1;
            float v_hat = new_v / bias_corr2;
            set_val(&l->biases, i, get_val(&l->biases, i) - opt->lr * m_hat / (sqrtf(v_hat) + opt->epsilon));
        }
    }
}
