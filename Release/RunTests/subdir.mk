################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../RunTests/main.cu 

CU_DEPS += \
./RunTests/main.d 

OBJS += \
./RunTests/main.o 


# Each subdirectory must supply rules for building sources it contributes
RunTests/%.o: ../RunTests/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_13,code=sm_13 -odir "RunTests" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_13,code=compute_13 -gencode arch=compute_13,code=sm_13  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


