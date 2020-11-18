// Copyright 2020 Kulandin Denis

#ifndef MODULES_TASK_2_KULANDIN_D_ITERATION_METHOD_ZEIDEL_ZEIDEL_H_
#define MODULES_TASK_2_KULANDIN_D_ITERATION_METHOD_ZEIDEL_ZEIDEL_H_

#include <vector>
#include <utility>

std::vector<double> randomVector(int sz);
bool сorrectMatrix(const std::vector<double>* matrix, int n);
std::pair<bool, std::vector<double>> zeidelSequential(std::vector<double>* a,
                                                      std::vector<double>* b,
                                                      size_t n, double eps);
std::pair<bool, std::vector<double>> zeidelParallel(std::vector<double>* a,
                                                    std::vector<double>* b,
                                                    size_t n, double eps);
void makeBeautifulMatrix(std::vector<double>* a, std::vector<double>* b, size_t n);

#endif  // MODULES_TASK_2_KULANDIN_D_ITERATION_METHOD_ZEIDEL_ZEIDEL_H_
