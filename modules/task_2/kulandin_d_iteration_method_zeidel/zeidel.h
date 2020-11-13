// Copyright 2020 Kulandin Denis
#include <vector>

bool сorrectMatrix(const std::vector<double>& matrix, int n);
std::pair<bool, std::vector<double>> zeidelSequential(std::vector<double>& a, std::vector<double>& b, size_t n, double eps);
std::pair<bool, std::vector<double>> zeidelParallel(std::vector<double>& a, std::vector<double>& b, size_t n, double eps);
