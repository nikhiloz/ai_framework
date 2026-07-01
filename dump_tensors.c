#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 1;
    
    uint32_t magic; fread(&magic, 4, 1, fp);
    uint32_t version; fread(&version, 4, 1, fp);
    uint64_t tensor_count; fread(&tensor_count, 8, 1, fp);
    uint64_t kv_count; fread(&kv_count, 8, 1, fp);
    
    // Skip KVs - this is brittle but just for quick inspection
    // For a real tool, I would parse it properly.
    // Given the previous debug output, I'll just skip the file pointer manually based on observed output
    // Actually, just listing names is fine if I know the structure.
    // Let's just assume I can't easily skip KVs without parsing.
    
    // Alternative: update gguf_loader.c to dump names it DOES find, 
    // and see what names are actually in the file.
    return 0;
}
