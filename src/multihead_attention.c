#include "multihead_attention.h"
#include <stdlib.h>
#include <stdbool.h>

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
    
    Matrix *head_outputs = (Matrix *)malloc(mha->num_heads * sizeof(Matrix));
    
    for (int h = 0; h < mha->num_heads; h++) {
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_input.data[i * head_dim + j] = input->data[i * embed_dim + (h * head_dim + j)];
            }
        }
        head_outputs[h] = attention_forward(&mha->heads[h], &head_input, true);
        free_matrix(&head_input);
    }
    
    Matrix concat = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < mha->num_heads; h++) {
            for (int j = 0; j < head_dim; j++) {
                concat.data[i * embed_dim + (h * head_dim + j)] = head_outputs[h].data[i * head_dim + j];
            }
        }
    }
    
    Matrix output = matrix_multiply(&concat, &mha->W_o);
    
    for (int h = 0; h < mha->num_heads; h++) free_matrix(&head_outputs[h]);
    free(head_outputs);
    free_matrix(&concat);
    
    return output;
}

Matrix mha_backward(MultiHeadAttention *mha, Matrix *input, Matrix *grad_output) {
    int seq_len = input->rows;
    int embed_dim = mha->embed_dim;
    int head_dim = mha->head_dim;
    float lr = 0.01f;

    Matrix Wo_T = matrix_transpose(&mha->W_o);
    Matrix dL_dConcat = matrix_multiply(grad_output, &Wo_T);

    Matrix *head_outputs = (Matrix *)malloc(mha->num_heads * sizeof(Matrix));
    for (int h = 0; h < mha->num_heads; h++) {
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_input.data[i * head_dim + j] = input->data[i * embed_dim + (h * head_dim + j)];
            }
        }
        head_outputs[h] = attention_forward(&mha->heads[h], &head_input, true);
        free_matrix(&head_input);
    }
    Matrix concat = create_matrix(seq_len, embed_dim);
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < mha->num_heads; h++) {
            for (int j = 0; j < head_dim; j++) {
                concat.data[i * embed_dim + (h * head_dim + j)] = head_outputs[h].data[i * head_dim + j];
            }
        }
    }
    Matrix concat_T = matrix_transpose(&concat);
    Matrix dL_dW_o = matrix_multiply(&concat_T, grad_output);
    for (int i = 0; i < embed_dim * embed_dim; i++) {
        mha->W_o.data[i] -= lr * dL_dW_o.data[i];
    }

    Matrix dL_dX = create_matrix(seq_len, embed_dim);
    matrix_fill_zero(&dL_dX);

    for (int h = 0; h < mha->num_heads; h++) {
        Matrix dL_dHeadOut = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                dL_dHeadOut.data[i * head_dim + j] = dL_dConcat.data[i * embed_dim + (h * head_dim + j)];
            }
        }
        Matrix head_input = create_matrix(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                head_input.data[i * head_dim + j] = input->data[i * embed_dim + (h * head_dim + j)];
            }
        }
        Matrix dL_dHeadIn = attention_backward(&mha->heads[h], &head_input, &dL_dHeadOut, true);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                dL_dX.data[i * embed_dim + (h * head_dim + j)] += dL_dHeadIn.data[i * head_dim + j];
            }
        }
        free_matrix(&dL_dHeadOut);
        free_matrix(&head_input);
        free_matrix(&dL_dHeadIn);
    }

    free_matrix(&Wo_T);
    free_matrix(&dL_dConcat);
    free_matrix(&concat);
    free_matrix(&concat_T);
    free_matrix(&dL_dW_o);
    for (int h = 0; h < mha->num_heads; h++) free_matrix(&head_outputs[h]);
    free(head_outputs);

    return dL_dX;
}
