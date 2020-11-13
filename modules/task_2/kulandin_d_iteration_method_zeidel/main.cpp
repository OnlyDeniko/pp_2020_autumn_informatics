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
    std::vector<double> a = {
        10, 1,  -1,
        1,  10, -1,
        -1, 1,  10
    };
    std::vector<double> b = {11, 10, 10};
    double eps = 0.000001;
    
    double gg = 1;
    for(auto &i : a) i = gg++;
    for(auto &i : b) i = gg++;
        //std::iota(a.begin(), a.end(), 1.);
        //std::iota(b.begin(), b.end(), n * n + 1);
    
    
    auto ans = zeidelParallel(a, b, n, eps);
    if (procRank == 0){
        std::cout << "RES = ";
        for(auto i : ans.second) std::cout << i << ' ';
        std::cout << '\n';
        auto seq = zeidelSequential(a, b, n, eps);
        std::cout << "SEQ = ";
        for(auto i : seq.second) std::cout << i << ' ';
        std::cout << '\n';
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
