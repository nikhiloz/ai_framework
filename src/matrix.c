#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)malloc(rows * cols * sizeof(float));
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
        m->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void matrix_fill_zero(Matrix *m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = 0.0f;
    }
}

void matrix_add(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void matrix_add_bias(Matrix *a, Matrix *bias, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i * a->cols + j] = a->data[i * a->cols + j] + bias->data[j];
        }
    }
}

void matrix_subtract(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void matrix_scalar_multiply(Matrix *m, float scalar) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = m->data[i] * scalar;
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
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

Matrix matrix_transpose(Matrix *m) {
    Matrix res = create_matrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            res.data[j * res.cols + i] = m->data[i * m->cols + j];
        }
    }
    return res;
}

void matrix_print(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        printf("[ ");
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", m->data[i * m->cols + j]);
        }
        printf("]\n");
    }
}

Matrix copy_matrix(Matrix *m) {
    Matrix res = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        res.data[i] = m->data[i];
    }
    return res;
}
