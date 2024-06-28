%%writefile gaussian_filter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define MASK_WIDTH 5
#define TILE_WIDTH 16
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
#define CHANNELS 3

__constant__ float mask[MASK_WIDTH * MASK_WIDTH];

__global__ void gaussianFilter(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH][CHANNELS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - MASK_WIDTH / 2;
    int col_i = col_o - MASK_WIDTH / 2;

    for (int c = 0; c < channels; ++c) {
        if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
            tile[ty][tx][c] = input[(row_i * width + col_i) * channels + c];
        } else {
            tile[ty][tx][c] = 0.0f;
        }
    }

    __syncthreads();

    float p_value[CHANNELS] = {0.0f};

    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; ++i) {
            for (int j = 0; j < MASK_WIDTH; ++j) {
                for (int c = 0; c < channels; ++c) {
                    p_value[c] += mask[i * MASK_WIDTH + j] * tile[i + ty][j + tx][c];
                }
            }
        }
        for (int c = 0; c < channels; ++c) {
            if (row_o < height && col_o < width) {
                output[(row_o * width + col_o) * channels + c] = (unsigned char) p_value[c];
            }
        }
    }
}

void initGaussianMask(float sigma) {
    float kernel[MASK_WIDTH * MASK_WIDTH];
    float sum = 0.0f;
    int half_width = MASK_WIDTH / 2;

    for (int i = -half_width; i <= half_width; ++i) {
        for (int j = -half_width; j <= half_width; ++j) {
            kernel[(i + half_width) * MASK_WIDTH + (j + half_width)] = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
            sum += kernel[(i + half_width) * MASK_WIDTH + (j + half_width)];
        }
    }

    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; ++i) {
        kernel[i] /= sum;
    }

    cudaMemcpyToSymbol(mask, kernel, MASK_WIDTH * MASK_WIDTH * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    int width, height, channels;
    unsigned char *input_image = stbi_load(input_filename, &width, &height, &channels, CHANNELS);
    if (input_image == NULL) {
        printf("Error loading the image: %s\n", input_filename);
        return 1;
    }

    size_t image_size = width * height * channels * sizeof(unsigned char);
    unsigned char *output_image = (unsigned char*)malloc(image_size);

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    cudaMemcpy(d_input, input_image, image_size, cudaMemcpyHostToDevice);

   
    initGaussianMask(1.0f);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1);

    gaussianFilter<<<dimGrid, dimBlock>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output_image, d_output, image_size, cudaMemcpyDeviceToHost);

    stbi_write_png(output_filename, width, height, channels, output_image, width * channels);

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(input_image);
    free(output_image);

    return 0;
}
