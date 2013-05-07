################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/helpers/shared_tests.cu 

CU_DEPS += \
./tests/helpers/shared_tests.d 

OBJS += \
./tests/helpers/shared_tests.o 


# Each subdirectory must supply rules for building sources it contributes
tests/helpers/%.o: ../tests/helpers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_13,code=sm_13 -odir "tests/helpers" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


