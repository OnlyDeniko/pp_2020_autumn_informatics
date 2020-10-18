// Copyright 2020 Luckyanchencko Ivan
#include"../../../modules/task_1/luckyanchencko_i_integral_rectangle/integral.h"
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <numeric>

float func1(float x) { return 1 / (log(x) + sin(pow(x, 2))); }
float func2(float x) { return (cos(x) + log(x)) / sin(x); }
float func3(float x) { return log(x) / (exp(pow(x, 3)) + cos(x)); }

float get_Paral_Integral(float (*func)(float), int a, int b) {
    int n = 10, ProcNum, ProcRank;
    float  h, ans = 0.0, res = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    h = (b - a) / static_cast<float>(n);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = ProcRank + 1; i <= n; i += ProcNum) {
        float x = (a + h * static_cast<float>(i)) - h / 2;
        res += func(x);
    }
    res = res * h;
    MPI_Reduce(&res, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return ans;
}

float get_Integral(float(*func)(float), int a, int b) {
    int n = 10;
    float  h, ans = 0.0, res = 0.0;
    h = (b - a) / static_cast<float>(n);
    for (int i =0; i <= n; i++) {
        float x = (a + h * static_cast<float>(i)) - h / 2;
        res += func(x);
    }
    res = res * h;
    return res;
}
