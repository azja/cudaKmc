#include "../headers/isinge.cuh"

 void kmcenergy::ising::prepare(float* e_I){

	cudaMemcpyToSymbol(_energy_I,e_I,sizeof(float) * N_ATOMS_ *N_ATOMS_);

 }

 void kmcenergy::ising::prepare(float* e_I, float r1){

	cudaMemcpyToSymbol(_energy_I,e_I,sizeof(float) * N_ATOMS_);
	cudaMemcpyToSymbol(&_r1,&r1,sizeof(float));
 }

 void kmcenergy::ising::prepare(float* e_I, float r1, float r2){

	cudaMemcpyToSymbol(_energy_I,e_I,sizeof(float) * N_ATOMS_);

	cudaMemcpyToSymbol(&_r1,&r1,sizeof(float));
	cudaMemcpyToSymbol(&_r2,&r2,sizeof(float));
 }

