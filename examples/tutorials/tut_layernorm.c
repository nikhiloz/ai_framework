#include "rmsnorm.h"
#include <stdio.h>

int main() {
    printf("--- RMS Normalization Test ---\n");
    
    int seq_len = 3;
    int dim = 4;
    
    Matrix input = create_matrix(seq_len, dim);
    // Fill with some values that have different means/variances
    float vals[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 20.0f, 30.0f, 40.0f,
        -1.0f, 0.0f, 1.0f, 2.0f
    };
    for(int i=0; i<seq_len * dim; i++) set_val(&input, i, vals[i]);
    
    printf("Original Input:\n");
    for(int i=0; i<seq_len; i++) {
        printf("Row %d: ", i);
        for(int j=0; j<dim; j++) printf("%.4f ", get_val(&input, i * dim + j));
        printf("\n");
    }
    
    RMSNorm ln = create_rmsnorm(dim);
    rmsnorm_forward(&ln, &input);
    
    printf("\nNormalized Input:\n");
    for(int i=0; i<seq_len; i++) {
        printf("Row %d: ", i);
        for(int j=0; j<dim; j++) printf("%.4f ", get_val(&input, i * dim + j));
        printf("\n");
    }
    
    free_matrix(&input);
    free_rmsnorm(&ln);
    
    return 0;
}
