#ifndef DATA_H
#define DATA_H

#include "matrix.h"
#include <stdlib.h>

typedef struct {
    Matrix features;
    Matrix targets;
    int num_samples;
} Dataset;

typedef struct {
    Dataset *dataset;
    int batch_size;
    int current_index;
    int num_batches;
    int *indices;
} DataLoader;

// Lifecycle
Dataset create_dataset(int num_samples, int feature_dim, int target_dim);
void free_dataset(Dataset *d);

DataLoader create_dataloader(Dataset *d, int batch_size, int shuffle);
void free_dataloader(DataLoader *dl);

// Data Access
void dataloader_next_batch(DataLoader *dl, Matrix *batch_features, Matrix *batch_targets);
void dataloader_reset(DataLoader *dl);

#endif
