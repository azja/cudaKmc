################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/builder/buildlattice.cu 

CU_DEPS += \
./tests/builder/buildlattice.d 

OBJS += \
./tests/builder/buildlattice.o 


# Each subdirectory must supply rules for building sources it contributes
tests/builder/%.o: ../tests/builder/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_13,code=sm_13 -odir "tests/builder" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


