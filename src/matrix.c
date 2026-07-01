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
    printf("DEBUG: create_matrix(%d, %d)\n", rows, cols);
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.precision = PRECISION_FLOAT32;
    m.quant = QUANT_NONE;
    m.is_mmaped = 0;
    m.data = malloc(rows * cols * sizeof(float));
    return m;
}

Matrix init_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.precision = PRECISION_FLOAT32;
    m.quant = QUANT_NONE;
    m.is_mmaped = 0;
    m.data = NULL;
    return m;
}

void free_matrix(Matrix *m) {
    if (m->data && !m->is_mmaped) {
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

// Helper to dequantize a Q4_0 element
float get_q4_0_val(Matrix *m, int i) {
    uint8_t *src = (uint8_t *)m->data;
    int block_size = 32;
    int block_bytes = 4 + 16; // 4 bytes scale, 16 bytes weights
    int b = i / block_size;
    int offset = i % block_size;
    
    float scale = *(float *)(src + b * block_bytes);
    uint8_t *weights = src + b * block_bytes + 4;
    uint8_t byte = weights[offset / 2];
    uint8_t q_val = (offset % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    
    return scale * (q_val - 8.0f);
}

float get_val(Matrix *m, int i) {
    if (!m->data) { fprintf(stderr, "DEBUG: NULL data access in get_val! Matrix(rows=%d, cols=%d)\n", m->rows, m->cols); abort(); }
    if (i < 0 || i >= m->rows * m->cols) {
        fprintf(stderr, "DEBUG: OOB access in get_val! Matrix(%dx%d), access index %d\n", m->rows, m->cols, i);
        abort();
    }
    if (m->quant == QUANT_Q4_0) {
        return get_q4_0_val(m, i);
    }
    if (m->precision == PRECISION_FLOAT32) {
        return ((float *)m->data)[i];
    } else {
        return float16_to_float(((uint16_t *)m->data)[i]);
    }
}

void set_val(Matrix *m, int i, float val) {
    if (!m->data) { fprintf(stderr, "DEBUG: NULL data access in set_val! Matrix(rows=%d, cols=%d)\n", m->rows, m->cols); abort(); }
    if (i < 0 || i >= m->rows * m->cols) {
        fprintf(stderr, "DEBUG: OOB access in set_val! Matrix(%dx%d), access index %d\n", m->rows, m->cols, i);
        abort();
    }
    if (m->precision == PRECISION_FLOAT32) {
        ((float *)m->data)[i] = val;
    } else {
        ((uint16_t *)m->data)[i] = float_to_float16(val);
    }
}

void matrix_add(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            int idx = i * a->stride + j;
            int idx_b = i * b->stride + j;
            int idx_res = i * result->stride + j;
            set_val(result, idx_res, get_val(a, idx) + get_val(b, idx_b));
        }
    }
}

void matrix_add_bias(Matrix *a, Matrix *bias, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            int idx_a = i * a->stride + j;
            int idx_bias = j; // Bias is usually (1, cols)
            int idx_res = i * result->stride + j;
            set_val(result, idx_res, get_val(a, idx_a) + get_val(bias, idx_bias));
        }
    }
}

void matrix_subtract(Matrix *a, Matrix *b, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            int idx_a = i * a->stride + j;
            int idx_b = i * b->stride + j;
            int idx_res = i * result->stride + j;
            set_val(result, idx_res, get_val(a, idx_a) - get_val(b, idx_b));
        }
    }
}

void matrix_scalar_multiply(Matrix *m, float scalar) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            int idx = i * m->stride + j;
            set_val(m, idx, get_val(m, idx) * scalar);
        }
    }
}

Matrix matrix_multiply(Matrix *a, Matrix *b) {
    if (!a || !b || !a->data || !b->data) { fprintf(stderr, "DEBUG: matrix_multiply input NULL data!\n"); abort(); }
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Matrix dimensions mismatch for multiplication\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = create_matrix(a->rows, b->cols);
    if (!result.data) {
        fprintf(stderr, "DEBUG: matrix_multiply failed to allocate result.data\n");
        abort();
    }
    matrix_fill_zero(&result);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += get_val(a, i * a->stride + k) * get_val(b, k * b->stride + j);
            }
            set_val(&result, i * result.stride + j, sum);
        }
    }
    printf("DEBUG: matrix_multiply success, result.data=%p\n", result.data);
    return result;
}

Matrix matrix_transpose(Matrix *m) {
    Matrix res = create_matrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            set_val(&res, j * res.stride + i, get_val(m, i * m->stride + j));
        }
    }
    return res;
}

void matrix_print(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        printf("[ ");
        for (int j = 0; j < m->cols; j++) {
            printf("%.2f ", get_val(m, i * m->stride + j));
        }
        printf("]\n");
    }
}

Matrix copy_matrix(Matrix *m) {
    Matrix res = create_matrix(m->rows, m->cols);
    res.precision = m->precision;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            set_val(&res, i * res.stride + j, get_val(m, i * m->stride + j));
        }
    }
    return res;
}
