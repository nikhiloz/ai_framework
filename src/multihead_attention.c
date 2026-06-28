#include "multihead_attention.h"
#include <stdlib.h>

MultiHeadAttention create_mha(int num_heads, int embed_dim) {
    MultiHeadAttention mha;
    mha.num_heads = num_heads;
    mha.embed_dim = embed_dim;
    mha.head_dim = embed_dim / num_heads;
    
    mha.heads = (AttentionLayer *)malloc(num_heads * sizeof(AttentionLayer));
    for (int i = 0; i < num_heads; i++) {
        mha.heads[i] = create_attention_layer(mha.head_dim);
    }
    
    mha.W_o = create_matrix(embed_dim, embed_dim);
    matrix_fill_random(&mha.W_o);
    
    return mha;
}

void free_mha(MultiHeadAttention *mha) {
    for (int i = 0; i < mha->num_heads; i++) {
        free_attention_layer(&mha->heads[i]);
    }
    free(mha->heads);
    free_matrix(&mha->W_o);
}

Matrix mha_forward(MultiHeadAttention *mha, Matrix *input) {
    int seq_len = input->rows;
    int embed_dim = mha->embed_dim;
    int head_dim = mha->head_dim;
    
    // 1. Split input into head_dim for each head
    // For a real implementation, we usually project the input into Q, K, V 
    // then split. Here, we simplify by splitting the input first and 
    // passing each slice to a head.
    
    Matrix *head_outputs = (Matrix *)malloc(mha->num_heads * sizeof(Matrix));
    
    for (int h = 0; h < mha->num_heads; h++) {
        // Extract slice for this head
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_input.data[i * head_dim + j] = input->data[i * embed_dim + (h * head_dim + j)];
            }
        }
        
        head_outputs[h] = attention_forward(&mha->heads[h], &head_input);
        free_matrix(&head_input);
    }
    
    // 2. Concatenate head outputs
    Matrix concat = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < mha->num_heads; h++) {
            for (int j = 0; j < head_dim; j++) {
                concat.data[i * embed_dim + (h * head_dim + j)] = head_outputs[h].data[i * head_dim + j];
            }
        }
    }
    
    // 3. Final linear projection
    Matrix output = matrix_multiply(&concat, &mha->W_o);
    
    // Cleanup
    for (int h = 0; h < mha->num_heads; h++) {
        free_matrix(&head_outputs[h]);
    }
    free(head_outputs);
    free_matrix(&concat);
    
    return output;
}
