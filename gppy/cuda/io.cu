// io.cu
#include "fitsio.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <cstdio>



void read_fits(const char* filename, std::vector<float>& data, long& width, long& height) {
    fitsfile* fptr;
    int status = 0, bitpix, naxis;
    long naxes[2] = {1, 1};

    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status);
    width = naxes[0];
    height = naxes[1];

    long npixels = width * height;
    data.resize(npixels);

    int anynul = 0;
    fits_read_img(fptr, TFLOAT, 1, npixels, nullptr, data.data(), &anynul, &status);
    if (anynul) {
        std::cerr << "Warning: Null pixels found in FITS image " << filename << std::endl;
    }

    fits_close_file(fptr, &status);

    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
}

void write_fits(const char* filename, const std::vector<float>& data, long width, long height) {
    fitsfile* fptr;
    int status = 0;
    long naxes[2] = {width, height};

    struct stat buffer;
    if (stat(filename, &buffer) == 0) {
        std::remove(filename);
    }

    // Create the FITS file
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    // Create the image HDU
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    // // Try to load header file with same base name
    // std::string head_name;
    // size_t pos = std::string(filename).rfind(".fits");
    // if (pos != std::string::npos) {
    //     head_name = std::string(filename).substr(0, pos) + ".header";
    // } else {
    //     head_name = std::string(filename) + ".header";  
    // }

    // std::ifstream head_file(head_name);
    // if (head_file.is_open()) {
    //     std::string line;
    //     while (std::getline(head_file, line)) {
    //         // FITS headers must be exactly 80 characters
    //         if (line.size() < 80) {
    //             line.append(80 - line.size(), ' ');
    //         } else if (line.size() > 80) {
    //             line = line.substr(0, 80);
    //         }

    //         fits_write_record(fptr, line.c_str(), &status);
    //         if (status) {
    //             fits_report_error(stderr, status);
    //             exit(status);
    //         }
    //     }
    //     head_file.close();
    // } else {
    //     std::cerr << "Warning: Header file not found: " << head_name << std::endl;
    // }

    // Write the image data
    fits_write_img(fptr, TFLOAT, 1, width * height, (void*)data.data(), &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    // Close the FITS file
    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
}