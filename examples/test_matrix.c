#include "../src/matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    printf("--- Testing AI Framework Matrix Engine ---\n");

    Matrix a = create_matrix(2, 3);
    Matrix b = create_matrix(3, 2);

    matrix_fill_random(&a);
    matrix_fill_random(&b);

    printf("Matrix A (2x3):\n");
    matrix_print(&a);
    printf("\nMatrix B (3x2):\n");
    matrix_print(&b);

    printf("\nPerforming Multiplication (A * B)...\n");
    Matrix res = matrix_multiply(&a, &b);

    printf("Result (2x2):\n");
    matrix_print(&res);

    printf("\nScaling result by 2.0...\n");
    matrix_scalar_multiply(&res, 2.0f);
    matrix_print(&res);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&res);

    printf("\nMatrix tests passed successfully!\n");
    return 0;
}
