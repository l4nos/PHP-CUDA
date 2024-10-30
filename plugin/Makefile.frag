# Object dependencies
CUDA_OBJECTS = \
	cuda_kernel.o \
	memory_pool.o \
	matrix_ops.o \
	conv_ops.o \
	cpu_ops.o \
	tensor_ops.o \
	neural_net.o \
	profiler.o

# Compilation rules
%.o: %.cu %.cuh cuda_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Main CUDA kernel compilation
cuda_kernel.o: $(wildcard *.cu) $(wildcard *.cuh)
	$(NVCC) $(NVCC_FLAGS) -c cuda_kernel.cu -o $@

# Clean rule
clean-cuda:
	rm -f $(CUDA_OBJECTS)

# Install rule
install-cuda: $(CUDA_OBJECTS)
	$(INSTALL) -m 755 $(CUDA_OBJECTS) $(EXTENSION_DIR)
