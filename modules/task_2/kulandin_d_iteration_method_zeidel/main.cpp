// Copyright 2020 Kulandin Denis
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "./zeidel.h"

const double EPS = 1e-8;

TEST(Parallel_MPI, Test_1) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    size_t n = 8;
    double eps = 1e-5;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a[0] = 61; a[1] = 2;  a[2] = 3;   a[3] = 4;   a[4] = 6;   a[5] = 8;   a[6] = 9;   a[7] = 2;
        a[8] = 2;  a[9] = 84; a[10] = 6;  a[11] = 4;  a[12] = 3;  a[13] = 2;  a[14] = 8;  a[15] = 7;
        a[16] = 3; a[17] = 6; a[18] = 68; a[19] = 2;  a[20] = 4;  a[21] = 3;  a[22] = 9;  a[23] = 1;
        a[24] = 4; a[25] = 4; a[26] = 2;  a[27] = 59; a[28] = 6;  a[29] = 4;  a[30] = 3;  a[31] = 8;
        a[32] = 6; a[33] = 3; a[34] = 4;  a[35] = 6;  a[36] = 93; a[36] = 8;  a[38] = 3;  a[39] = 1;
        a[40] = 8; a[41] = 2; a[42] = 3;  a[43] = 4;  a[44] = 8;  a[45] = 98; a[46] = 3;  a[47] = 1;
        a[48] = 9; a[49] = 8; a[50] = 9;  a[51] = 3;  a[52] = 3;  a[53] = 3;  a[54] = 85; a[55] = 7;
        a[56] = 2; a[57] = 7; a[58] = 1;  a[59] = 8;  a[60] = 1;  a[61] = 1;  a[62] = 7;  a[63] = 98;

        b[0] = 95; b[1] = 116; b[2] = 96; b[3] = 90; b[4] = 124; b[5] = 127; b[6] = 127; b[7] = 125;
        makeBeautifulMatrix(&a, &b, n);
    }

    double timeParallel = MPI_Wtime();
    auto ans = zeidelParallel(&a, &b, n, eps);
    timeParallel = MPI_Wtime() - timeParallel;

    if (!procRank) {
        double timeSequential = MPI_Wtime();
        auto seq = zeidelSequential(&a, &b, n, eps);
        timeSequential = MPI_Wtime() - timeSequential;
        ASSERT_EQ(ans.first, seq.first);
        if (!ans.first) return;
        double mse = 0;
        for (size_t i = 0; i < n; ++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, EPS);
        for (size_t i = 0; i < n; ++i) {
            double su = 0;
            for (size_t j = 0; j < n; ++j) {
                su += a[i * n + j] * ans.second[j];
            }
            ASSERT_LE(fabs(su - b[i]), sqrt(eps) * n);
        }
        printf("Parallel time:              %.10f\n", timeParallel);
        printf("Sequential time:            %.10f\n", timeSequential);
    }
}

TEST(Parallel_MPI, Test_2) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    size_t n = 3;
    double eps = 1e-6;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a = {
            10, 1, -1,
            1, 10, -1,
            -1, 1, 10
        };
        b = {11, 10, 10};
        makeBeautifulMatrix(&a, &b, n);
    }

    double timeParallel = MPI_Wtime();
    auto ans = zeidelParallel(&a, &b, n, eps);
    timeParallel = MPI_Wtime() - timeParallel;

    if (!procRank) {
        double timeSequential = MPI_Wtime();
        auto seq = zeidelSequential(&a, &b, n, eps);
        timeSequential = MPI_Wtime() - timeSequential;
        ASSERT_EQ(ans.first, seq.first);
        if (!ans.first) return;
        double mse = 0;
        for (size_t i = 0; i < n; ++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, EPS);
        for (size_t i = 0; i < n; ++i) {
            double su = 0;
            for (size_t j = 0; j < n; ++j) {
                su += a[i * n + j] * ans.second[j];
            }
            ASSERT_LE(fabs(su - b[i]), sqrt(eps) * n);
        }
        printf("Parallel time:              %.10f\n", timeParallel);
        printf("Sequential time:            %.10f\n", timeSequential);
    }
}

TEST(Parallel_MPI, Test_3) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    size_t n = 2;
    double eps = 1e-6;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a = {
            4, 2,
            1, 3
        };
        b = {1, -1};
        makeBeautifulMatrix(&a, &b, n);
    }

    double timeParallel = MPI_Wtime();
    auto ans = zeidelParallel(&a, &b, n, eps);
    timeParallel = MPI_Wtime() - timeParallel;

    if (!procRank) {
        double timeSequential = MPI_Wtime();
        auto seq = zeidelSequential(&a, &b, n, eps);
        timeSequential = MPI_Wtime() - timeSequential;
        ASSERT_EQ(ans.first, seq.first);
        if (!ans.first) return;
        double mse = 0;
        for (size_t i = 0; i < n; ++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, EPS);
        for (size_t i = 0; i < n; ++i) {
            double su = 0;
            for (size_t j = 0; j < n; ++j) {
                su += a[i * n + j] * ans.second[j];
            }
            ASSERT_LE(fabs(su - b[i]), sqrt(eps) * n);
        }
        printf("Parallel time:              %.10f\n", timeParallel);
        printf("Sequential time:            %.10f\n", timeSequential);
    }
}

TEST(Parallel_MPI, Test_5) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    size_t n = 3;
    double eps = 1e-6;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a = {
            115, -20, -75,
            15, -50, -5,
            6, 2, 20
        };
        b = {20, -40, 28};
        makeBeautifulMatrix(&a, &b, n);
    }

    double timeParallel = MPI_Wtime();
    auto ans = zeidelParallel(&a, &b, n, eps);
    timeParallel = MPI_Wtime() - timeParallel;

    if (!procRank) {
        double timeSequential = MPI_Wtime();
        auto seq = zeidelSequential(&a, &b, n, eps);
        timeSequential = MPI_Wtime() - timeSequential;
        ASSERT_EQ(ans.first, seq.first);
        if (!ans.first) return;
        double mse = 0;
        for (size_t i = 0; i < n; ++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, EPS);
        for (size_t i = 0; i < n; ++i) {
            double su = 0;
            for (size_t j = 0; j < n; ++j) {
                su += a[i * n + j] * ans.second[j];
            }
            ASSERT_LE(fabs(su - b[i]), sqrt(eps) * n);
        }
        printf("Parallel time:              %.10f\n", timeParallel);
        printf("Sequential time:            %.10f\n", timeSequential);
    }
}

TEST(Parallel_MPI, Test_4) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    size_t n = 3;
    double eps = 1e-6;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        b = {5, 5, 5};
        makeBeautifulMatrix(&a, &b, n);
    }

    double timeParallel = MPI_Wtime();
    auto ans = zeidelParallel(&a, &b, n, eps);
    timeParallel = MPI_Wtime() - timeParallel;

    if (!procRank) {
        double timeSequential = MPI_Wtime();
        auto seq = zeidelSequential(&a, &b, n, eps);
        timeSequential = MPI_Wtime() - timeSequential;
        ASSERT_EQ(ans.first, seq.first);
        if (!ans.first) return;
        double mse = 0;
        for (size_t i = 0; i < n; ++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, EPS);
        for (size_t i = 0; i < n; ++i) {
            double su = 0;
            for (size_t j = 0; j < n; ++j) {
                su += a[i * n + j] * ans.second[j];
            }
            ASSERT_LE(fabs(su - b[i]), sqrt(eps) * n);
        }
        printf("Parallel time:              %.10f\n", timeParallel);
        printf("Sequential time:            %.10f\n", timeSequential);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
