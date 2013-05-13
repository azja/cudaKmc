#include <cstdlib>
#include <vector>
#include <gtest/gtest.h>

#include "kernels.cuh"


class Set1DArrayTest : public ::testing::Test
{
protected:
    Set1DArrayTest() {}
    virtual ~Set1DArrayTest() {}
    virtual void SetUp()
    {
        cudaDeviceReset();
        std::srand(time(0));
    }
    virtual void TearDown() {}
protected:

    void CreateSample(std::vector<int>& sample)
    {
        for (size_t i = 0; i < sample.size(); ++i) sample[i] = rand() % 1024;
    }
};


TEST_F(Set1DArrayTest, RegularNumElementsZeroIndex)
{
    const int numElements = 512;
    const int index = 0;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

TEST_F(Set1DArrayTest, RegularNumElementsLastIndex)
{
    const int numElements = 512;
    const int index = numElements - 1;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

TEST_F(Set1DArrayTest, RegularNumElementsRandomIndex)
{
    const int numElements = 512;
    const int index = rand() % numElements;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

TEST_F(Set1DArrayTest, UnregularNumElementsZeroIndex)
{
    const int numElements = 1 + rand() % 4096;
    const int index = 0;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

TEST_F(Set1DArrayTest, UnregularNumElementsLastIndex)
{
    const int numElements =  1 + rand() % 4096;
    const int index = numElements - 1;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

TEST_F(Set1DArrayTest, UnregularNumElementsRandomIndex)
{
    const int numElements = 1 + rand() % 4096;
    const int index = rand() % numElements;
    const int value = rand() % 1024;

    std::vector<int> h_sample(numElements);
    ASSERT_EQ(h_sample.size(), numElements);
    CreateSample(h_sample);

    int* d_sample;
    ASSERT_EQ(cudaMalloc((void**) &d_sample, sizeof(int) * numElements), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_sample, h_sample.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice), cudaSuccess);

    dim3 threads(128);
    dim3 blocks((numElements + threads.x - 1)/threads.x);
    set1dArrayElement<<<blocks, threads>>>(index, value, d_sample, numElements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int result;
    EXPECT_EQ(cudaMemcpy(&result, &d_sample[index], sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(result, value);
    EXPECT_EQ(cudaFree(d_sample), cudaSuccess);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
