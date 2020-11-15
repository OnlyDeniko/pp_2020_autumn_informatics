// Copyright 2020 Kulandin Denis
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include "./zeidel.h"

TEST(Parallel_MPI, Test_1) {
    int procNum, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    //std::cout << procRank + 1 << ' ' << procNum << '\n';
    size_t n = 3;
    double eps = 0.000001;
    std::vector<double> a(n * n, 1);
    std::vector<double> b(n, 1);
    if (!procRank) {
        a = {
            2, 1, 1,
            1, -1, 0,
            3, -1, 2
        };
        b = {
            2, -2, 2
        };
        // double gg = 1;
        // for(auto &i : a) i = gg++;
        // for(auto &i : b) i = gg++;
        makeBeautifulMatrix(a, b, n);
        std::cout << "!\n";
        for(auto i : a) std::cout << i << ' ';
        std::cout << '\n';
    }
    
    auto ans = zeidelParallel(a, b, n, eps);
    if (procRank == 0){
        // std::cout << "RES = !" << ans.first << "! ";
        // for(auto i : ans.second) std::cout << i << ' ';
        // std::cout << '\n';
        auto seq = zeidelSequential(a, b, n, eps);
        //std::cout << "SEQ = !" << seq.first << "! ";
        //for(auto i : seq.second) std::cout << i << ' ';
        //std::cout << '\n';
        ASSERT_EQ(ans.first, seq.first);
        double mse = 0;
        for(size_t i = 0;i < n;++i) {
            mse += (ans.second[i] - seq.second[i]) * (ans.second[i] - seq.second[i]);
        }
        mse = sqrt(mse);
        ASSERT_LE(mse, 1e-8);
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
