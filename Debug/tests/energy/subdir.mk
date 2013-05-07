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
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "tests/energy" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


