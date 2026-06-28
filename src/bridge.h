#ifndef BRIDGE_H
#define BRIDGE_H

// Opaque handle to the TransformerModel
typedef void* ModelHandle;

ModelHandle create_model_bridge(int vocab_size, int embed_dim, int num_heads, int num_blocks, int ffn_dim);
void free_model_bridge(ModelHandle handle);

// Generates a single next token given a prompt string
int generate_single_token_bridge(ModelHandle handle, const char* prompt);

// Decodes a token ID back to a character
char* decode_token_bridge(ModelHandle handle, int token_id);

#endif
