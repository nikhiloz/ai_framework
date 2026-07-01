#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 1;

    uint32_t magic; fread(&magic, 4, 1, fp);
    uint32_t version; fread(&version, 4, 1, fp);
    uint64_t tensor_count; fread(&tensor_count, 8, 1, fp);
    uint64_t kv_count; fread(&kv_count, 8, 1, fp);

    printf("Magic: 0x%X, Version: %u, Tensors: %lu, KVs: %lu\n", magic, version, tensor_count, kv_count);

    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t key_len; fread(&key_len, 8, 1, fp);
        char *key = malloc(key_len + 1);
        fread(key, 1, key_len, fp);
        key[key_len] = '\0';
        
        uint32_t val_type; fread(&val_type, 4, 1, fp);
        
        printf("KV %lu: Key: %s, Type: %u\n", i, key, val_type);
        
        // Skip value
        if (val_type == 8) { // string
             uint64_t len; fread(&len, 8, 1, fp);
             fseek(fp, len, SEEK_CUR);
        } else if (val_type == 9) { // array
            uint64_t len; fread(&len, 8, 1, fp);
            uint32_t elem_type; fread(&elem_type, 4, 1, fp);
            // This is still fragile, but let's see if we can get past this.
            // For now, let's just assume we can skip fixed sizes or break if too hard.
            // Actually, let's try just printing the key and breaking if it's an array to see if we get the next key.
            printf("  -> Array len: %lu, type: %u\n", len, elem_type);
            // This will definitely crash on complex arrays, but let's try.
        } else {
             // Scalar - skip
             static const int scalar_sizes[] = {1,1,2,2,4,4,4,8,0,0,8,8,8}; // types 0-12
             fseek(fp, scalar_sizes[val_type], SEEK_CUR);
        }
        free(key);
    }
    
    fclose(fp);
    return 0;
}
