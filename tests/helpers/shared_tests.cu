#include "shared_tests.cuh"
#include "../../headers/utils.h"
#include "../../macros/errors.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>

static __global__ void func(int* in,int *out, int size) {

	extern __shared__ int X[];
    int i,tile;


	__syncthreads();

	for( i = 0, tile = 0; i < size; i += blockDim.x,++tile){
		X[threadIdx.x] = in[tile * blockDim.x + threadIdx.x];
		__syncthreads();
		for(int j = 0;j < blockDim.x; ++j){
			if(tile * blockDim.x + j <size){
			if(X[j] > 1)
		      out[tile*blockDim.x + threadIdx.x] += X[j];
			}
		}
		__syncthreads();
	}

}







const int BLOCK_SIZE = 512;

struct comparer {
	__device__ int operator()(int x,int y) {
		return x==y?1:0;
	}
};

namespace tests {
 namespace helpers {

void test(int size) {
	int* in;
	int* out;

    CHECK_ERROR(cudaMalloc((void**)&in,sizeof(int)*size));
    CHECK_ERROR(cudaMalloc((void**)&out,sizeof(int)*size));

    thrust::device_ptr<int> d_in= thrust::device_pointer_cast(in);
    thrust::device_ptr<int> d_out= thrust::device_pointer_cast(out);


    thrust::sequence(d_in,d_in + size);
    thrust::fill(d_out,d_out + size,0);



    func<<<(size - 1)/BLOCK_SIZE +1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(in,out,size);
    cudaDeviceSynchronize();

    int result = thrust::inner_product(d_in,d_in +size,d_out,0,thrust::plus<int>(),comparer());


    std::cout<<"Result = "<<result<<std::endl;

    cudaFree(in);
    cudaFree(out);


}
}
}
