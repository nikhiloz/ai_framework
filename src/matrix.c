#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Helper to convert float to float16 (simplified)
uint16_t float_to_float16(float f) {
    // Basic conversion logic (needs proper IEEE 754 conversion)
    // For now, cast to 16-bit to preserve structure.
    return (uint16_t)(f);
}

// Helper to convert float16 to float
float float16_to_float(uint16_t f16) {
    return (float)(f16);
}

Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.precision = PRECISION_FLOAT32;
    m.data = malloc(rows * cols * sizeof(float));
    return m;
}

void free_matrix(Matrix *m) {
    if (m->data) {
        free(m->data);
        m->data = NULL;
    }
}

void matrix_fill_random(Matrix *m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        float val = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        if (m->precision == PRECISION_FLOAT32) {
            ((float *)m->data)[i] = val;
        } else {
            ((uint16_t *)m->data)[i] = float_to_float16(val);
        }
    }
}

void matrix_fill_zero(Matrix *m) {
    memset(m->data, 0, m->rows * m->cols * (m->precision == PRECISION_FLOAT32 ? sizeof(float) : sizeof(uint16_t)));
}

float get_val(Matrix *m, int i) {
    if (m->precision == PRECISION_FLOAT32) {
        return ((float *)m->data)[i];
    } else {
        return float16_to_float(((uint16_t *)m->data)[i]);
    }
}

void set_val(Matrix *m, int i, float val) {
    if (m->precision == PRECISION_FLOAT32) {
        ((float *)m->data)[i] = val;
    } else {
        ((uint16_t *)m->data)[i] = float_to_float16(val);
    }
}

void matrix_add(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        set_val(result, i, get_val(a, i) + get_val(b, i));
    }
}

void matrix_add_bias(Matrix *a, Matrix *bias, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            set_val(result, i * a->cols + j, get_val(a, i * a->cols + j) + get_val(bias, j));
        }
    }
}

void matrix_subtract(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        set_val(result, i, get_val(a, i) - get_val(b, i));
    }
}

void matrix_scalar_multiply(Matrix *m, float scalar) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        set_val(m, i, get_val(m, i) * scalar);
    }
}

Matrix matrix_multiply(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Matrix dimensions mismatch for multiplication\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = create_matrix(a->rows, b->cols);
    matrix_fill_zero(&result);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += get_val(a, i * a->cols + k) * get_val(b, k * b->cols + j);
            }
            set_val(&result, i * result.cols + j, sum);
        }
    }
    return result;
}

Matrix matrix_transpose(Matrix *m) {
    Matrix res = create_matrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            set_val(&res, j * res.cols + i, get_val(m, i * m->cols + j));
        }
    }
    return res;
}

void matrix_print(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        printf("[ ");
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", get_val(m, i * m->cols + j));
        }
        printf("]\n");
    }
}

Matrix copy_matrix(Matrix *m) {
    Matrix res = create_matrix(m->rows, m->cols);
    res.precision = m->precision;
    for (int i = 0; i < m->rows * m->cols; i++) {
        set_val(&res, i, get_val(m, i));
    }
    return res;
}
