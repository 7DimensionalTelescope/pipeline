// io.cu
#include "fitsio.h"
#include "io.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


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

    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
    
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    std::string fits_name(filename);
    std::string clean_name = fits_name;

    if (!clean_name.empty() && clean_name[0] == '!') {
        clean_name = clean_name.substr(1);
    }

    std::string head_name;
    size_t pos = clean_name.rfind(".fits");
    if (pos != std::string::npos) {
        head_name = clean_name.substr(0, pos) + ".header";
    } else {
        head_name = clean_name + ".header";  
    }

    std::ifstream head_file(head_name);
    if (head_file.is_open()) {
        std::string line;
        while (std::getline(head_file, line)) {
            fits_write_record(fptr, line.c_str(), &status);
            if (status) {
                fits_report_error(stderr, status);
                exit(status);
            }
        }
        head_file.close();
    } else {
        std::cerr << "Warning: Header file not found: " << head_name << std::endl;
    }

    fits_write_img(fptr, TFLOAT, 1, width * height, (void*)data.data(), &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
}