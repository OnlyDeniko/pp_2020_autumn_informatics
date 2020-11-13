// Copyright 2020 Kulandin Denis
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <numeric>
#include <iostream>
#include <algorithm>
#include "../../../modules/task_2/kulandin_d_iteration_method_zeidel/zeidel.h"

const double EPS = 1e-8;

bool converge(const std::vector<double> &x, const std::vector<double>& last, double eps){
    double ans = 0;
    for(size_t i = 0;i < x.size();++i) {
        double tmp = x[i] - last[i];
        ans += tmp * tmp;
    }
    return sqrt(ans) < eps;
}

bool сorrectMatrix(const std::vector<double>& matrix, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (matrix[i * n + i] == 0) {
            return false;
        }
        double sum = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                sum += fabs(matrix[i * n + j]);
            }
        }
        sum /= fabs(matrix[i * n + i]);
        if (sum > 1. + EPS) {
            return false;
        }
    }
    return true;
}

void swap(std::vector<double>& a, size_t n, size_t row1, size_t row2){
    for(size_t i = 0;i < n;++i){
        std::swap(a[row1 * n + i], a[row2 * n + i]);
    }    
}

double calcParallel(std::vector<double>& a, std::vector<double>& x, size_t row, size_t n){
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    std::cout << procRank + 1 << ' ' << row << '\n';
    size_t offset = n / procNum;
    std::vector<int> sendCounts(procNum);
    std::vector<int> displs(procNum, row * n);
    std::vector<int> displsX(procNum, 0);
    std::vector<double> recvBufA(offset + n % procNum, 0);
    std::vector<double> recvBufX(offset + n % procNum, 0);
    if (procRank == 0){
        size_t rem = n;
        for (size_t i = 0;i < procNum;++i) {
            sendCounts[i] = (i == procNum - 1 ? rem : offset);
            rem -= offset;
        }
        for (size_t i = 1;i < procNum;++i) {
            displs[i]   = displs[i - 1] + sendCounts[i - 1];
            displsX[i]  = displsX[i - 1] + sendCounts[i - 1];
        }
        MPI_Scatterv(a.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, recvBufA.data(), recvBufA.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(x.data(), sendCounts.data(), displsX.data(), MPI_DOUBLE, recvBufX.data(), recvBufX.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    for(auto i : recvBufA) std::cout << "! " << procRank + 1 << ' ' << i << '\n';
    for(auto i : recvBufX) std::cout << "!! " << procRank + 1 << ' ' << i << '\n';
    double localAns = 0;
    double globalAns = 0;
    for (size_t i = 0;i < recvBufA.size();++i) {
        localAns += recvBufA[i] * recvBufX[i];
    }
    //MPI_Gatherv(recvBufA.data(), recvBufA.size(), MPI_DOUBLE, a.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(recvBufX.data(), recvBufX.size(), MPI_DOUBLE, x.data(), sendCounts.data(), displsX.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localAns, &globalAns, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return globalAns;
}

std::pair<bool, std::vector<double>> zeidelSequential(std::vector<double>& a, std::vector<double>& b, size_t n, double eps) {
    for(size_t j = 0;j < n;++j){
        double ma = fabs(a[j * n + j]);
        size_t ind = j;
        for(size_t i = j + 1;i < n;++i) {
            if (ma < fabs(a[i * n + j]) - EPS) {
                ma = fabs(a[i * n + j]);
                ind = i;                           
            }
        }
        swap(a, n, ind, j);
        std::swap(b[ind], b[j]);
    }

    // for(auto i : a) std::cout << i << ' ';
    // std::cout << '\n';
    // for(auto i : b) std::cout << i << ' ';
    // std::cout << '\n';
    int cntIterations = 100;
    std::vector<double> x(n, 0), last(n, 0);
    do{
        last = x;
        for(size_t i = 0;i < n;++i){
            x[i] = 0;
            double gg = 0;
            for(size_t j = 0;j < n;++j) {
                gg += a[i * n + j] * x[j];
            }            
            x[i] = (b[i] - gg) / a[i * n + i];
        }
    }while(!converge(x, last, eps) && cntIterations--);
    return std::make_pair(cntIterations != 0, x);
}

std::pair<bool, std::vector<double>> zeidelParallel(std::vector<double>& a, std::vector<double>& b, size_t n, double eps) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    for(size_t j = 0;j < n;++j){
        double ma = fabs(a[j * n + j]);
        size_t ind = j;
        for(size_t i = j + 1;i < n;++i) {
            if (ma < fabs(a[i * n + j]) - EPS) {
                ma = fabs(a[i * n + j]);
                ind = i;                           
            }
        }
        swap(a, n, ind, j);
        std::swap(b[ind], b[j]);
    }
    
    
    int cntIterations = 100;
    std::vector<double> x(n, 0), last(n, 0);
    do{
        last = x;
        for(size_t i = 0;i < n;++i){
            x[i] = 0;
            MPI_Barrier(MPI_COMM_WORLD);
            double gg = calcParallel(a, x, i, n);
            MPI_Barrier(MPI_COMM_WORLD);
            x[i] = (b[i] - gg) / a[i * n + i];
            std::cout << "rank = " << procRank + 1 << " i = " << i << " x[i] = " << x[i] << '\n';
        }
        if (cntIterations == 99) return std::make_pair(1, x);
        
    }while(!converge(x, last, eps) && cntIterations--);

    return std::make_pair(cntIterations != 0, x);
}