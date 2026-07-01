#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "sampling.h"
#include "positional.h"

// ... [create_model, free_model, model_forward, model_generate, model_train_step remain unchanged] ...

int load_llama_weights_mmap(TransformerModel *model, const char *filename) {
    printf("DEBUG: Entering load_llama_weights_mmap for file: %s\n", filename);
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open file for mmap");
        return -1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Failed to get file size");
        close(fd);
        return -1;
    }

    size_t file_size = sb.st_size;
    float *map = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return -1;
    }

    int *int_map = (int *)map;
    int v_size = int_map[0];
    int e_dim = int_map[1];
    int n_blocks = int_map[2];

    if (v_size != model->vocab_size || e_dim != model->embed_dim || n_blocks != model->num_blocks) {
        fprintf(stderr, "Model architecture mismatch during load\n");
        munmap(map, file_size);
        close(fd);
        return -1;
    }

    // Pointer to start of float data (after 3 ints)
    float *data = map + 3;

    // Load embedding
    size_t embed_size = model->embedding.weights.rows * model->embedding.weights.cols;
    memcpy(model->embedding.weights.data, data, embed_size * sizeof(float));
    data += embed_size;

    // Load transformer blocks
    for (int i = 0; i < model->num_blocks; i++) {
        TransformerBlock *tb = &model->blocks[i];
        
        // MHA Weights
        size_t q_size = tb->mha.W_q.rows * tb->mha.W_q.cols;
        memcpy(tb->mha.W_q.data, data, q_size * sizeof(float));
        data += q_size;
        
        size_t k_size = tb->mha.W_k.rows * tb->mha.W_k.cols;
        memcpy(tb->mha.W_k.data, data, k_size * sizeof(float));
        data += k_size;
        
        size_t v_size = tb->mha.W_v.rows * tb->mha.W_v.cols;
        memcpy(tb->mha.W_v.data, data, v_size * sizeof(float));
        data += v_size;
        
        size_t o_size = tb->mha.W_o.rows * tb->mha.W_o.cols;
        memcpy(tb->mha.W_o.data, data, o_size * sizeof(float));
        data += o_size;

        // LN1
        size_t ln1_size = tb->ln1.weight.rows * tb->ln1.weight.cols;
        memcpy(tb->ln1.weight.data, data, ln1_size * sizeof(float));
        data += ln1_size;

        // FFN
        size_t w1_size = tb->W1.rows * tb->W1.cols;
        memcpy(tb->W1.data, data, w1_size * sizeof(float));
        data += w1_size;
        
        size_t w3_size = tb->W3.rows * tb->W3.cols;
        memcpy(tb->W3.data, data, w3_size * sizeof(float));
        data += w3_size;
        
        size_t b1_size = tb->b1.rows * tb->b1.cols;
        memcpy(tb->b1.data, data, b1_size * sizeof(float));
        data += b1_size;
        
        size_t w2_size = tb->W2.rows * tb->W2.cols;
        memcpy(tb->W2.data, data, w2_size * sizeof(float));
        data += w2_size;
        
        size_t b2_size = tb->b2.rows * tb->b2.cols;
        memcpy(tb->b2.data, data, b2_size * sizeof(float));
        data += b2_size;

        // LN2
        size_t ln2_size = tb->ln2.weight.rows * tb->ln2.weight.cols;
        memcpy(tb->ln2.weight.data, data, ln2_size * sizeof(float));
        data += ln2_size;
    }

    // Load Prediction Head
    size_t wo_size = model->head.W_out.rows * model->head.W_out.cols;
    memcpy(model->head.W_out.data, data, wo_size * sizeof(float));
    data += wo_size;
    
    size_t bo_size = model->head.b_out.rows * model->head.b_out.cols;
    memcpy(model->head.b_out.data, data, bo_size * sizeof(float));
    data += bo_size;

    munmap(map, file_size);
    close(fd);
    return 0;
}
