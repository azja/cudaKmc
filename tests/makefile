INC=-I/../headers -I/kernels
BASE_SRC_PATH = ../src
BASE_HEAD_PATH = ../headers 
TEST_SRC_PATH_KERNELS = kernels
TEST_SRC_PATH_ENERGY =  energy
TEST_SRC_PATH_BUILDER = builder
TEST_SRC_PATH_HELPERS = helpers
OPT =  -O3  -arch compute_13 -Xcompiler -fopenmp -lgomp 

ifeq ($(CFG),debug)
OPT +=  -DDEBUG
endif

kmctest: utils.o anames.o kernels.o cpukernels.o sample.o sampled.o runtests.o isinge.o findneigbours.o simulationrta.o ising1.o  buildlattice.o transparams.o shared_tests.o
	nvcc ${OPT} kernels.o anames.o cpukernels.o sample.o sampled.o isinge.o findneigbours.o utils.o simulationrta.o ising1.o buildlattice.o transparams.o shared_tests.o  runtests.o -o kmctest
kernels.o: ${BASE_SRC_PATH}/kernels.cu 
	nvcc -c ${OPT} ${BASE_SRC_PATH}/kernels.cu
utils.o: ${BASE_SRC_PATH}/utils.cu
	nvcc -c ${OPT} ${BASE_SRC_PATH}/utils.cu
anames.o: ${BASE_SRC_PATH}/anames.cu
	nvcc -c ${OPT} ${BASE_SRC_PATH}/anames.cu
cpukernels.o: ${BASE_SRC_PATH}/cpukernels.cu
	nvcc -c ${OPT} ${BASE_SRC_PATH}/cpukernels.cu
sample.o: ${BASE_SRC_PATH}/sample.cu
	nvcc -c ${OPT} ${BASE_SRC_PATH}/sample.cu
sampled.o: ${BASE_SRC_PATH}/sampled.cu 
	nvcc -c ${OPT} ${BASE_SRC_PATH}/sampled.cu
isinge.o: ${BASE_SRC_PATH}/isinge.cu 
	nvcc -c ${OPT} ${BASE_SRC_PATH}/isinge.cu
findneigbours.o: ${TEST_SRC_PATH_KERNELS}/findneigbours.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_KERNELS}/findneigbours.cu
simulationrta.o: ${TEST_SRC_PATH_KERNELS}/simulationrta.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_KERNELS}/simulationrta.cu
ising1.o: ${TEST_SRC_PATH_ENERGY}/ising1.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_ENERGY}/ising1.cu
buildlattice.o: ${TEST_SRC_PATH_BUILDER}/buildlattice.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_BUILDER}/buildlattice.cu
shared_tests.o: ${TEST_SRC_PATH_HELPERS}/shared_tests.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_HELPERS}/shared_tests.cu
transparams.o: ${TEST_SRC_PATH_KERNELS}/transparams.cu
	nvcc -c ${OPT} ${TEST_SRC_PATH_KERNELS}/transparams.cu
runtests.o: runtests.cu
	nvcc -c ${OPT} runtests.cu
clean:
	rm *.o
        
