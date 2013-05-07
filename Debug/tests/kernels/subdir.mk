################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/kernels/findneigbours.cu \
../tests/kernels/simulationrta.cu \
../tests/kernels/transparams.cu 

CU_DEPS += \
./tests/kernels/findneigbours.d \
./tests/kernels/simulationrta.d \
./tests/kernels/transparams.d 

OBJS += \
./tests/kernels/findneigbours.o \
./tests/kernels/simulationrta.o \
./tests/kernels/transparams.o 


# Each subdirectory must supply rules for building sources it contributes
tests/kernels/%.o: ../tests/kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "tests/kernels" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


