#ifndef MODEL_H
#define MODEL_H

#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "transformer_block.h"
#include "prediction.h"

typedef struct {
    Tokenizer tokenizer;
    Embedding embedding;
    int num_blocks;
    TransformerBlock *blocks;
    PredictionHead head;
    int embed_dim;
    int vocab_size;
} TransformerModel;

TransformerModel create_model(int vocab_size, int embed_dim, int num_heads, int num_blocks, int ffn_dim);
void free_model(TransformerModel *model);

Matrix model_forward(TransformerModel *model, int *tokens, int seq_len);
int model_generate(TransformerModel *model, const char *prompt, int max_len);
void model_train_step(TransformerModel *model, int *tokens, int *targets, int seq_len, float lr);

#endif
