################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../tests/energy/ising1.cu 

CU_DEPS += \
./tests/energy/ising1.d 

OBJS += \
./tests/energy/ising1.o 


# Each subdirectory must supply rules for building sources it contributes
tests/energy/%.o: ../tests/energy/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_13,code=sm_13 -odir "tests/energy" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


