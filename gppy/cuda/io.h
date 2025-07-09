// io.h
#pragma once
#include <vector>
#include <string>

void read_fits(const char* filename, std::vector<float>& data, long& width, long& height);
void write_fits(const char* filename, const std::vector<float>& data, long width, long height);
