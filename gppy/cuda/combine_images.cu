#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include "io.h"

#define checkCudaError(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Device kernel: subtract subtract_img from each image in stack
__global__ void subtract_kernel(float* d_stack, const float* d_subtract, int N, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * size;
    if (idx >= total) return;
    int pixel_idx = idx % size;
    d_stack[idx] -= d_subtract[pixel_idx];
}

// Device kernel: divide each image by its median scalar
__global__ void normalize_kernel(float* d_stack, const float* d_medians, int N, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * size) return;
    int img_idx = idx / size;
    d_stack[idx] /= d_medians[img_idx];
}

// Compute median per pixel across images (N values per pixel)
// Assumes N <= 64 (adjust bubble sort accordingly)
__global__ void median_kernel(const float* d_stack, float* d_median, int N, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Allocate shared memory for sorting N values per pixel
    extern __shared__ float sdata[];

    // Load values for this pixel from all images into shared mem
    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        sdata[i] = d_stack[i * size + idx];
    }
    __syncthreads();

    // Simple bubble sort (inefficient but okay for small N)
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N - i - 1; ++j) {
            if (sdata[j] > sdata[j + 1]) {
                float tmp = sdata[j];
                sdata[j] = sdata[j + 1];
                sdata[j + 1] = tmp;
            }
        }
    }
    __syncthreads();

    // Write median
    if (N % 2 == 1)
        d_median[idx] = sdata[N / 2];
    else
        d_median[idx] = 0.5f * (sdata[N / 2 - 1] + sdata[N / 2]);
}

// Compute stddev per pixel across images with ddof=1 using median as center
__global__ void stddev_kernel(const float* d_stack, const float* d_median, float* d_stddev, int N, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float mean_sq_diff = 0.f;
    float median_val = d_median[idx];
    for (int i = 0; i < N; ++i) {
        float diff = d_stack[i * size + idx] - median_val;
        mean_sq_diff += diff * diff;
    }
    if (N > 1)
        d_stddev[idx] = sqrtf(mean_sq_diff / (N - 1));
    else
        d_stddev[idx] = 0.f;
}

void compute_per_image_medians(const float* d_stack, float* d_medians, int N, int size) {
    // Copy each image from d_stack to host, median on CPU (since median per image is just 1D)
    // Alternatively implement GPU median per image; here CPU for simplicity

    std::vector<float> temp(size);
    for (int i = 0; i < N; ++i) {
        checkCudaError(cudaMemcpy(temp.data(), d_stack + i * size, size * sizeof(float), cudaMemcpyDeviceToHost));
        std::nth_element(temp.begin(), temp.begin() + size/2, temp.end());
        float median = (size % 2 == 1) ? temp[size/2] : 0.5f * (temp[size/2-1] + temp[size/2]);
        checkCudaError(cudaMemcpy(d_medians + i, &median, sizeof(float), cudaMemcpyHostToDevice));
    }
}

void combine_images(
    const std::vector<std::string>& image_paths,
    std::vector<float>& out_median,
    std::vector<float>& out_stddev,
    int device_id = 0,
    const std::string& subtract_path = "",
    bool normalize = false
) {
    cudaSetDevice(device_id);

    // Load first image to get size
    long w, h;
    std::vector<float> temp;
    read_fits(image_paths[0].c_str(), temp, w, h);
    int size = w * h;
    int N = image_paths.size();

    // Allocate host & device memory for stack
    float* d_stack;
    checkCudaError(cudaMalloc(&d_stack, N * size * sizeof(float)));

    // Load all images to device stack
    for (int i = 0; i < N; ++i) {
        std::vector<float> img;
        long iw, ih;
        read_fits(image_paths[i].c_str(), img, iw, ih);
        if (iw != w || ih != h) {
            std::cerr << "Image size mismatch\n";
            exit(EXIT_FAILURE);
        }
        checkCudaError(cudaMemcpy(d_stack + i * size, img.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Optional subtract
    if (!subtract_path.empty()) {
        std::vector<float> subtract_img;
        long sw, sh;
        read_fits(subtract_path.c_str(), subtract_img, sw, sh);
        if (sw != w || sh != h) {
            std::cerr << "Subtract image size mismatch\n";
            exit(EXIT_FAILURE);
        }
        float* d_subtract;
        checkCudaError(cudaMalloc(&d_subtract, size * sizeof(float)));
        checkCudaError(cudaMemcpy(d_subtract, subtract_img.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        int total = N * size;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        subtract_kernel<<<gridSize, blockSize>>>(d_stack, d_subtract, N, size);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        cudaFree(d_subtract);
    }

    // Optional normalize (divide each image by its median)
    if (normalize) {
        float* d_medians;
        checkCudaError(cudaMalloc(&d_medians, N * sizeof(float)));

        compute_per_image_medians(d_stack, d_medians, N, size);

        int total = N * size;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        normalize_kernel<<<gridSize, blockSize>>>(d_stack, d_medians, N, size);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());

        cudaFree(d_medians);
    }

    // Allocate median and stddev on device
    float* d_median;
    float* d_stddev;
    checkCudaError(cudaMalloc(&d_median, size * sizeof(float)));
    checkCudaError(cudaMalloc(&d_stddev, size * sizeof(float)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t shared_mem_size = N * sizeof(float); // shared mem per block for median kernel

    median_kernel<<<gridSize, blockSize, shared_mem_size>>>(d_stack, d_median, N, size);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    stddev_kernel<<<gridSize, blockSize>>>(d_stack, d_median, d_stddev, N, size);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    // Copy median and stddev back to host
    out_median.resize(size);
    out_stddev.resize(size);
    checkCudaError(cudaMemcpy(out_median.data(), d_median, size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(out_stddev.data(), d_stddev, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_stack);
    cudaFree(d_median);
    cudaFree(d_stddev);
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " -input imgs... -median_out median.fits -std_out std.fits [-subtract file] [-norm] [-device id]" << std::endl;
        return 1;
    }

    std::vector<std::string> image_paths;
    std::string median_out, std_out;
    std::string subtract_path = "";
    bool normalize = false;
    int device_id = 0;

    for (int i = 1; i < argc;) {
        std::string arg = argv[i];

        if (arg == "-input") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                image_paths.emplace_back(argv[i++]);
            }
        } else if (arg == "-median_out" && i + 1 < argc) {
            median_out = argv[++i];
            ++i;
        } else if (arg == "-std_out" && i + 1 < argc) {
            std_out = argv[++i];
            ++i;
        } else if (arg == "-subtract" && i + 1 < argc) {
            subtract_path = argv[++i];
            ++i;
        } else if (arg == "-norm") {
            normalize = true;
            ++i;
        } else if (arg == "-device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
            ++i;
        } else {
            std::cerr << "Unknown or malformed option: " << arg << std::endl;
            return 2;
        }
    }

    if (image_paths.empty() || median_out.empty() || std_out.empty()) {
        std::cerr << "Must specify -input, -median_out, and -std_out" << std::endl;
        return 3;
    }

    std::vector<float> median_img, std_img;

    combine_images(image_paths, median_img, std_img, device_id, subtract_path, normalize);

    // Get image shape from first image
    long w, h;
    std::vector<float> tmp;
    read_fits(image_paths[0].c_str(), tmp, w, h);

    // Save results
    write_fits(median_out.c_str(), median_img, w, h);
    write_fits(std_out.c_str(), std_img, w, h);

    return 0;
}