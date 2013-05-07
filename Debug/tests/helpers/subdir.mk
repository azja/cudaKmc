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
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "tests/helpers" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


