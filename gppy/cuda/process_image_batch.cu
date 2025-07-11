
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstring>
#include "io.h"  // FITS read/write helper


#define checkCudaError(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }


__global__ void reduction_kernel(float* images, const float* bias, const float* dark, const float* flat, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float b = bias[idx];
    float d = dark[idx];
    float f = flat[idx];

    for (int i = 0; i < batch_size; ++i) {
        float* img = images + i * size;
        img[idx] = (img[idx] - b - d) / f;
    }
}

void process_images(
    const std::vector<std::string>& image_paths,
    const std::string& bias_path,
    const std::string& dark_path,
    const std::string& flat_path,
    std::vector<std::string>& output_paths,
    int device_id = 0
) {
    cudaSetDevice(device_id);

    const int batch_size = 20;
    long w = 0, h = 0;

    // Read calibration images into vectors and get width and height from bias image
    std::vector<float> bias, dark, flat;
    read_fits(bias_path.c_str(), bias, w, h);
    read_fits(dark_path.c_str(), dark, w, h);
    read_fits(flat_path.c_str(), flat, w, h);

    int size = w * h;
    float *d_bias, *d_dark, *d_flat, *d_images;
    checkCudaError(cudaMalloc(&d_bias, size * sizeof(float)));
    checkCudaError(cudaMalloc(&d_dark, size * sizeof(float)));
    checkCudaError(cudaMalloc(&d_flat, size * sizeof(float)));
    checkCudaError(cudaMemcpy(d_bias, bias.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_dark, dark.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_flat, flat.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    float* pinned_input;
    float* pinned_output;
    cudaHostAlloc((void**)&pinned_input, batch_size * size * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&pinned_output, batch_size * size * sizeof(float), cudaHostAllocDefault);
    checkCudaError(cudaMalloc(&d_images, batch_size * size * sizeof(float)));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
        int current_batch = std::min(batch_size, static_cast<int>(image_paths.size() - i));

        // Read batch of images into temporary vectors, then memcpy to pinned input buffer
        for (int j = 0; j < current_batch; ++j) {
            long iw = 0, ih = 0;
            std::vector<float> temp_img;
            read_fits(image_paths[i + j].c_str(), temp_img, iw, ih);
            if (iw != w || ih != h) {
                std::cerr << "Image size mismatch!" << std::endl;
                exit(EXIT_FAILURE);
            }
            std::memcpy(pinned_input + j * size, temp_img.data(), size * sizeof(float));
        }

        checkCudaError(cudaMemcpyAsync(d_images, pinned_input, current_batch * size * sizeof(float), cudaMemcpyHostToDevice, stream));

        int blockSize = 1024;
        int gridSize = (size + blockSize - 1) / blockSize;
        reduction_kernel<<<gridSize, blockSize, 0, stream>>>(d_images, d_bias, d_dark, d_flat, size, current_batch);

        checkCudaError(cudaMemcpyAsync(pinned_output, d_images, current_batch * size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Write batch results from pinned output buffer slices wrapped as vectors
        for (int j = 0; j < current_batch; ++j) {
            std::filesystem::create_directories(std::filesystem::path(output_paths[i + j]).parent_path());
            std::vector<float> output_vec(pinned_output + j * size, pinned_output + (j + 1) * size);
            write_fits(output_paths[i + j].c_str(), output_vec, w, h);
        }
    }

    cudaFree(d_bias);
    cudaFree(d_dark);
    cudaFree(d_flat);
    cudaFree(d_images);
    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    cudaStreamDestroy(stream);
}


int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " -input imgs... -output outs... -bias file -dark file -flat file [-overwrite]" << std::endl;
        return 1;
    }

    std::vector<std::string> image_paths;
    std::vector<std::string> output_paths;
    std::string bias_path, dark_path, flat_path;
    int device_id = 0; 

    for (int i = 1; i < argc;) {
        std::string arg = argv[i];

        if (arg == "-input") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                image_paths.emplace_back(argv[i++]);
            }
        } else if (arg == "-output") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                output_paths.emplace_back(argv[i++]);
            }
        } else if (arg == "-bias" && i + 1 < argc) {
            bias_path = argv[++i];
            ++i;
        } else if (arg == "-dark" && i + 1 < argc) {
            dark_path = argv[++i];
            ++i;
        } else if (arg == "-flat" && i + 1 < argc) {
            flat_path = argv[++i];
            ++i;
        } else if (arg == "-device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
            ++i;
        } else {
            std::cerr << "Unknown or malformed option: " << arg << std::endl;
            return 2;
        }
    }

    if (image_paths.empty() || output_paths.empty()) {
        std::cerr << "Error: -input and -output are required." << std::endl;
        return 3;
    }
    if (image_paths.size() != output_paths.size()) {
        std::cerr << "Error: The number of input and output files must match." << std::endl;
        return 4;
    }
    if (bias_path.empty() || dark_path.empty() || flat_path.empty()) {
        std::cerr << "Error: -bias, -dark, and -flat must all be provided." << std::endl;
        return 5;
    }

    process_images(image_paths, bias_path, dark_path, flat_path, output_paths, device_id);
    return 0;
}
