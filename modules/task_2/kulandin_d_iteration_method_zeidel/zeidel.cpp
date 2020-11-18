// Copyright 2020 Kulandin Denis
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <random>
#include <numeric>
#include <iostream>
#include <utility>
#include "../../../modules/task_2/kulandin_d_iteration_method_zeidel/zeidel.h"

const double EPS = 1e-8;

std::vector<double> randomVector(int sz) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<double> ans(sz);
    for (int i = 0; i < sz; ++i) {
        ans[i] = gen();
    }
    return ans;
}

bool converge(const std::vector<double>* x, const std::vector<double>* last, double eps) {
    double ans = 0;
    for (size_t i = 0; i < (*x).size(); ++i) {
        double tmp = (*x)[i] - (*last)[i];
        ans += tmp * tmp;
    }
    return sqrt(ans) < eps;
}

bool correctMatrix(const std::vector<double>* matrix, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double su = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j != i) su += (*matrix)[i * n + j];
        }
        if (su + EPS > (*matrix)[i * n + i]) {
            return false;
        }
    }
    return true;
}

void swap(std::vector<double>* a, size_t n, size_t row1, size_t row2) {
    for (size_t i = 0; i < n; ++i) {
        std::swap((*a)[row1 * n + i], (*a)[row2 * n + i]);
    }
}

void makeBeautifulMatrix(std::vector<double>* a, std::vector<double>* b, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        double ma = fabs((*a)[j * n + j]);
        size_t ind = j;
        for (size_t i = j + 1; i < n; ++i) {
            if (ma < fabs((*a)[i * n + j]) - EPS) {
                ma = fabs((*a)[i * n + j]);
                ind = i;
            }
        }
        swap(a, n, ind, j);
        std::swap((*b)[ind], (*b)[j]);
    }
}

std::pair<bool, std::vector<double> > zeidelSequential(std::vector<double>* a, std::vector<double>* b,
                                                       size_t n, double eps) {
    int cntIterations = 100;
    std::vector<double> x(n, 0), last(n, 0);
    do {
        last = x;
        for (size_t i = 0; i < n; ++i) {
            x[i] = 0;
            double gg = 0;
            for (size_t j = 0; j < n; ++j) {
                gg += (*a)[i * n + j] * x[j];
            }
            x[i] = ((*b)[i] - gg) / (*a)[i * n + i];
        }
    } while (!converge(&x, &last, eps) && cntIterations--);
    return std::make_pair(cntIterations != 0, x);
}

double calcParallel(std::vector<double>* a, std::vector<double>* x, size_t row, size_t n) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    const size_t delta = n / procNum;
    const size_t rem = n % procNum;
    if (procRank == 0) {
        for (int j = 1; j < procNum; ++j) {
            MPI_Send(&(*a)[0] + row * n + rem + j * delta, delta, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
            MPI_Send(&(*x)[0] + rem + j * delta, delta, MPI_DOUBLE, j, 1, MPI_COMM_WORLD);
        }
    }
    std::vector<double> local(procRank == 0 ? rem + delta : delta, 0);
    std::vector<double> localX(procRank == 0 ? rem + delta : delta, 0);
    if (procRank == 0) {
        local = std::vector<double>((*a).begin() + row * n, (*a).begin() + row * n + delta + rem);
        localX = std::vector<double>((*x).begin(), (*x).begin() + delta + rem);
    } else {
        MPI_Status status;
        MPI_Recv(&local[0], delta, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&localX[0], delta, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
    double localSum = 0;
    double globalSum = 0;
    for (size_t i = 0; i < (procRank == 0 ? rem + delta : delta); ++i) {
        localSum += local[i] * localX[i];
    }
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return globalSum;
}

std::pair<bool, std::vector<double> > zeidelParallel(std::vector<double>* a, std::vector<double>* b,
                                                     size_t n, double eps) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    char check = !correctMatrix(a, n);
    MPI_Bcast(&check, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    int cntIterations = 100;
    std::vector<double> x(n, 0), last(n, 0);
    char result;
    do {
        last = x;
        for (size_t i = 0; i < n; ++i) {
            x[i] = 0;
            double gg = calcParallel(a, &x, i, n);
            x[i] = ((*b)[i] - gg) / (*a)[i * n + i];
        }
        if (!converge(&x, &last, eps) && cntIterations--) {
            result = 1;
        } else {
            result = 0;
        }
        MPI_Bcast(&result, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    } while (result);

    return std::make_pair(cntIterations != 0, x);
}
