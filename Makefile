CC = gcc
CFLAGS = -Isrc -Wall -Wextra -O2
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin
TUT_DIR = examples/tutorials

# Source files
SRCS = $(SRC_DIR)/matrix.c $(SRC_DIR)/nn.c $(SRC_DIR)/optimizer.c $(SRC_DIR)/data.c $(SRC_DIR)/tokenizer.c $(SRC_DIR)/embedding.c $(SRC_DIR)/positional.c $(SRC_DIR)/attention.c $(SRC_DIR)/multihead_attention.c $(SRC_DIR)/layernorm.c $(SRC_DIR)/transformer_block.c
OBJS = $(SRCS:.c=.o)

# Tutorials to build
TUTS = $(wildcard $(TUT_DIR)/*.c)
TUT_BINS = $(patsubst $(TUT_DIR)/%.c, $(BIN_DIR)/tut_%, $(TUTS))

all: $(BIN_DIR) $(TUT_BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build tutorials
$(BIN_DIR)/tut_%: $(TUT_DIR)/%.c $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Build objects
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR) $(OBJS)

.PHONY: all clean
