#include "gguf_loader.h"
#include "model.h"
#include "transformer_block.h"
#include "embedding.h"
#include "prediction.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747

GGUF_TensorInfo* find_tensor(GGUF_ModelInfo* info, const char* name) {
    if (!info) return NULL;
    for (uint64_t i = 0; i < info->tensor_count; i++) {
        if (strcmp(info->tensors[i].name, name) == 0) return &info->tensors[i];
    }
    return NULL;
}

uint64_t read_gguf_value(FILE *fp, uint32_t type) {
    uint64_t val = 0;
    switch (type) {
        case 0: { uint8_t v; if (fread(&v, 1, 1, fp) == 1) val = v; break; }
        case 1: { int8_t v; if (fread(&v, 1, 1, fp) == 1) val = (uint8_t)v; break; }
        case 2: { uint16_t v; if (fread(&v, 2, 1, fp) == 1) val = v; break; }
        case 3: { int16_t v; if (fread(&v, 2, 1, fp) == 1) val = (uint16_t)v; break; }
        case 4: { uint32_t v; if (fread(&v, 4, 1, fp) == 1) val = v; break; }
        case 5: { int32_t v; if (fread(&v, 4, 1, fp) == 1) val = (uint32_t)v; break; }
        case 6: { float v; if (fread(&v, 4, 1, fp) == 1) val = *(uint32_t*)&v; break; }
        case 7: { double v; if (fread(&v, 8, 1, fp) == 1) val = *(uint64_t*)&v; break; }
        case 8: { 
            uint64_t len;
            if (fread(&len, 8, 1, fp) != 1) return 0;
            char *s = malloc(len + 1);
            if (fread(s, 1, len, fp) != len) { free(s); return 0; }
            s[len] = '\0';
            free(s);
            return 0;
        }
        case 9: { return 0; }
        case 10: { uint64_t v; if (fread(&v, 8, 1, fp) == 1) val = v; break; }
        case 11: { int64_t v; if (fread(&v, 8, 1, fp) == 1) val = (uint64_t)v; break; }
        default: break;
    }
    return val;
}

GGUF_ModelInfo* gguf_parse_header(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    GGUF_ModelInfo *info = calloc(1, sizeof(GGUF_ModelInfo));
    fread(&info->magic, 4, 1, fp);
    fread(&info->version, 4, 1, fp);
    fread(&info->tensor_count, 8, 1, fp);
    fread(&info->kv_pair_count, 8, 1, fp);
    
    for (uint64_t i = 0; i < info->kv_pair_count; i++) {
        uint64_t key_len;
        fread(&key_len, 8, 1, fp);
        char *key = malloc(key_len + 1);
        fread(key, 1, key_len, fp);
        key[key_len] = '\0';
        uint32_t val_type;
        fread(&val_type, 4, 1, fp);
        
        if (val_type == 9) { // Array
            uint32_t element_type;
            uint64_t array_len;
            fread(&element_type, 4, 1, fp);
            fread(&array_len, 8, 1, fp);
            if (element_type == 8) {
                for (uint64_t j = 0; j < array_len; j++) {
                    uint64_t s_len;
                    fread(&s_len, 8, 1, fp);
                    fseek(fp, s_len, SEEK_CUR);
                }
            } else {
                uint64_t elem_size = 8;
                switch (element_type) {
                    case 0: case 1: elem_size = 1; break;
                    case 2: case 3: elem_size = 2; break;
                    case 4: case 5: case 6: elem_size = 4; break;
                    case 7: case 10: case 11: case 12: elem_size = 8; break;
                    default: elem_size = 8;
                }
                fseek(fp, array_len * elem_size, SEEK_CUR);
            }
        } else {
            uint64_t val = read_gguf_value(fp, val_type);
            if (strcmp(key, "llama.embedding_length") == 0) info->embed_dim = (int)val;
            else if (strcmp(key, "llama.block_count") == 0) info->num_blocks = (int)val;
            else if (strcmp(key, "llama.attention.head_count") == 0) info->num_heads = (int)val;
            else if (strcmp(key, "llama.attention.head_dim") == 0) info->head_dim = (int)val;
            else if (strcmp(key, "llama.feed_forward_length") == 0) info->ffn_dim = (int)val;
            else if (strcmp(key, "llama.vocab_size") == 0) info->vocab_size = (int)val;
        }
        free(key);
    }
    
    info->tensors = malloc(info->tensor_count * sizeof(GGUF_TensorInfo));
    for (uint64_t i = 0; i < info->tensor_count; i++) {
        GGUF_TensorInfo *t = &info->tensors[i];
        uint64_t name_len;
        fread(&name_len, 8, 1, fp);
        char *name = malloc(name_len + 1);
        fread(name, 1, name_len, fp);
        name[name_len] = '\0';
        strncpy(t->name, name, 255);
        free(name);
        
        uint32_t ndims;
        fread(&ndims, 4, 1, fp);
        uint64_t *dims = malloc(ndims * sizeof(uint64_t));
        t->elements = 1;
        for (uint64_t j = 0; j < ndims; j++) {
            fread(&dims[j], 8, 1, fp);
            t->elements *= dims[j];
        }
        free(dims);
        
        uint32_t type;
        fread(&type, 4, 1, fp);
        t->type = (GGUF_TensorType)type;
        fread(&t->offset, 8, 1, fp);
    }
    
    info->tensor_data_offset = ftell(fp);
    fclose(fp);
    return info;
}

void gguf_free_model_info(GGUF_ModelInfo* info) {
    if (!info) return;
    if (info->tensors) free(info->tensors);
    free(info);
}

static void set_matrix_from_tensor(Matrix *m, GGUF_TensorInfo *t, void *mapped_data) {
    if (!t) return;
    
    // Free existing memory if it was allocated and not mmapped
    if (m->data && !m->is_mmaped) {
        free(m->data);
    }
    
    // GGUF tensor offsets are relative to the start of the data section.
    m->data = (uint8_t*)mapped_data + t->offset;
    m->is_mmaped = 1;
    m->precision = PRECISION_FLOAT32;
    m->quant = QUANT_NONE;
}

TransformerModel load_model_gguf(const char* filename) {
    GGUF_ModelInfo *info = gguf_parse_header(filename);
    if (!info) exit(1);
    
    TransformerModel model = init_model(info->vocab_size, info->embed_dim, info->num_heads, info->num_blocks, info->ffn_dim);
    
    int fd = open(filename, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    
    void *map_base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (map_base == MAP_FAILED) { perror("mmap"); exit(1); }
    
    void *mapped_data = (uint8_t*)map_base + info->tensor_data_offset;
    
    model.mmap_ptr = map_base;
    model.mmap_size = st.st_size;
    
    set_matrix_from_tensor(&model.embedding.weights, find_tensor(info, "token_embd.weight"), mapped_data);
    for (int i = 0; i < model.num_blocks; i++) {
        TransformerBlock *tb = &model.blocks[i];
        char name[256];
        sprintf(name, "blk.%d.attn_q.weight", i); set_matrix_from_tensor(&tb->mha.W_q, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.attn_k.weight", i); set_matrix_from_tensor(&tb->mha.W_k, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.attn_v.weight", i); set_matrix_from_tensor(&tb->mha.W_v, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.attn_output.weight", i); set_matrix_from_tensor(&tb->mha.W_o, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.attn_norm.weight", i); set_matrix_from_tensor(&tb->ln1.weight, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.ffn_gate.weight", i); set_matrix_from_tensor(&tb->W1, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.ffn_gate.bias", i);
        GGUF_TensorInfo *t_b1 = find_tensor(info, name);
        if (t_b1) {
            set_matrix_from_tensor(&tb->b1, t_b1, mapped_data);
        } else {
            tb->b1 = create_matrix(1, info->ffn_dim);
            matrix_fill_zero(&tb->b1);
        }
        
        sprintf(name, "blk.%d.ffn_up.weight", i); set_matrix_from_tensor(&tb->W3, find_tensor(info, name), mapped_data);
        sprintf(name, "blk.%d.ffn_down.weight", i); set_matrix_from_tensor(&tb->W2, find_tensor(info, name), mapped_data);
        
        sprintf(name, "blk.%d.ffn_down.bias", i);
        GGUF_TensorInfo *t_b2 = find_tensor(info, name);
        if (t_b2) {
            set_matrix_from_tensor(&tb->b2, t_b2, mapped_data);
        } else {
            tb->b2 = create_matrix(1, info->embed_dim);
            matrix_fill_zero(&tb->b2);
        }
        
        sprintf(name, "blk.%d.ffn_norm.weight", i); set_matrix_from_tensor(&tb->ln2.weight, find_tensor(info, name), mapped_data);
    }
    set_matrix_from_tensor(&model.head.W_out, find_tensor(info, "output.weight"), mapped_data);
    
    gguf_free_model_info(info);
    return model;
}
