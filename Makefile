CC = gcc
CFLAGS = -Isrc -Wall -Wextra -O2 -fPIC
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin
TUT_DIR = examples/tutorials
LIB_NAME = libaiframework.so

# Source files
SRCS = $(SRC_DIR)/matrix.c $(SRC_DIR)/nn.c $(SRC_DIR)/optimizer.c $(SRC_DIR)/data.c $(SRC_DIR)/tokenizer.c $(SRC_DIR)/embedding.c $(SRC_DIR)/positional.c $(SRC_DIR)/attention.c $(SRC_DIR)/multihead_attention.c $(SRC_DIR)/rmsnorm.c $(SRC_DIR)/transformer_block.c $(SRC_DIR)/prediction.c $(SRC_DIR)/model.c $(SRC_DIR)/model_mmap.c $(SRC_DIR)/sampling.c $(SRC_DIR)/bridge.c $(SRC_DIR)/chat_data.c
OBJS = $(SRCS:.c=.o)

# Tutorials to build
TUTS = $(wildcard $(TUT_DIR)/*.c)
TUT_BINS = $(patsubst $(TUT_DIR)/%.c, $(BIN_DIR)/tut_%, $(TUTS))

all: $(BIN_DIR) $(TUT_BINS) $(LIB_NAME)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build shared library
$(LIB_NAME): $(OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# Build tutorials
$(BIN_DIR)/tut_%: $(TUT_DIR)/%.c $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# Build objects
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN_DIR) $(OBJS) $(LIB_NAME)

.PHONY: all clean
