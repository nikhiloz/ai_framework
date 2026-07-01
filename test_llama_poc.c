#include <stdio.h>
#include <stdlib.h>
#include "../../llama.cpp/include/llama.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }

    printf("Model loaded successfully: %s\n", model_path);
    printf("Number of parameters: %llu\n", llama_model_n_params(model));

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
