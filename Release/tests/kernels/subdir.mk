################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/kernels/findneigbours.cu \
../tests/kernels/transparams.cu 

CU_DEPS += \
./tests/kernels/findneigbours.d \
./tests/kernels/transparams.d 

OBJS += \
./tests/kernels/findneigbours.o \
./tests/kernels/transparams.o 


# Each subdirectory must supply rules for building sources it contributes
tests/kernels/%.o: ../tests/kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_13,code=sm_13 -odir "tests/kernels" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


