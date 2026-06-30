#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Dataset create_dataset(int num_samples, int feature_dim, int target_dim) {
    Dataset d;
    d.num_samples = num_samples;
    d.features = create_matrix(num_samples, feature_dim);
    d.targets = create_matrix(num_samples, target_dim);
    return d;
}

void free_dataset(Dataset *d) {
    free_matrix(&d->features);
    free_matrix(&d->targets);
}

DataLoader create_dataloader(Dataset *d, int batch_size, int shuffle) {
    DataLoader dl;
    dl.dataset = d;
    dl.batch_size = batch_size;
    dl.current_index = 0;
    dl.num_batches = (d->num_samples + batch_size - 1) / batch_size;
    dl.indices = (int *)malloc(d->num_samples * sizeof(int));

    for (int i = 0; i < d->num_samples; i++) {
        dl.indices[i] = i;
    }

    if (shuffle) {
        for (int i = d->num_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = dl.indices[i];
            dl.indices[i] = dl.indices[j];
            dl.indices[j] = temp;
        }
    }

    return dl;
}

void free_dataloader(DataLoader *dl) {
    free(dl->indices);
}

void dataloader_next_batch(DataLoader *dl, Matrix *batch_features, Matrix *batch_targets) {
    int start = dl->current_index;
    int end = (start + dl->batch_size < dl->dataset->num_samples) ? 
               (start + dl->batch_size) : dl->dataset->num_samples;
    int actual_batch_size = end - start;

    batch_features->rows = actual_batch_size;
    batch_targets->rows = actual_batch_size;

    for (int i = 0; i < actual_batch_size; i++) {
        int idx = dl->indices[start + i];
        
        // Copy features
        for (int j = 0; j < dl->dataset->features.cols; j++) {
            set_val(batch_features, i * batch_features->cols + j, 
                    get_val(&dl->dataset->features, idx * dl->dataset->features.cols + j));
        }
        
        // Copy targets
        for (int j = 0; j < dl->dataset->targets.cols; j++) {
            set_val(batch_targets, i * batch_targets->cols + j, 
                    get_val(&dl->dataset->targets, idx * dl->dataset->targets.cols + j));
        }
    }

    dl->current_index = end;
}

void dataloader_reset(DataLoader *dl) {
    dl->current_index = 0;
}
