#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdint.h>

typedef enum {
    PRECISION_FLOAT32,
    PRECISION_FLOAT16
} Precision;

typedef enum {
    QUANT_NONE,
    QUANT_Q4_0,
    QUANT_Q4_K
} QuantType;

typedef struct {
    int rows;
    int cols;
    int stride;
    Precision precision;
    QuantType quant;
    void *data;
    int is_mmaped;
} Matrix;

// Lifecycle
Matrix create_matrix(int rows, int cols);
Matrix init_matrix(int rows, int cols);
void free_matrix(Matrix *m);

// Basic Operations
void matrix_fill_random(Matrix *m);
void matrix_fill_zero(Matrix *m);
void matrix_add(Matrix *a, Matrix *b, Matrix *result);
void matrix_add_bias(Matrix *a, Matrix *bias, Matrix *result);
void matrix_subtract(Matrix *a, Matrix *b, Matrix *result);
void matrix_scalar_multiply(Matrix *m, float scalar);

// The Core of AI: Matrix Multiplication
Matrix matrix_multiply(Matrix *a, Matrix *b);
Matrix matrix_transpose(Matrix *m);
Matrix copy_matrix(Matrix *m);

// Utility
void matrix_print(Matrix *m);
float get_val(Matrix *m, int i);
void set_val(Matrix *m, int i, float val);

#endif
