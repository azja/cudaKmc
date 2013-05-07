################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/anames.cu \
../src/cpukernels.cu \
../src/isinge.cu \
../src/kernels.cu \
../src/main.cu \
../src/sample.cu \
../src/sampled.cu \
../src/utils.cu 

CU_DEPS += \
./src/anames.d \
./src/cpukernels.d \
./src/isinge.d \
./src/kernels.d \
./src/main.d \
./src/sample.d \
./src/sampled.d \
./src/utils.d 

OBJS += \
./src/anames.o \
./src/cpukernels.o \
./src/isinge.o \
./src/kernels.o \
./src/main.o \
./src/sample.o \
./src/sampled.o \
./src/utils.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


