# PHP CUDA Extension

A near production-ready PHP extension that provides CUDA support for high-performance computing and deep learning tasks. It still needs a lot more testing, anyone who can test or wants to contribute, feel free. I want nothing in return for writing this the goal was merely to level the playing field between PHP and Python and prove that PHP is as capable as it's reptilian counterpart.

That being said, any donations are never refused.

## Features

### Core CUDA Support
- Device Management
  - Get CUDA device count
  - Query device properties
  - Set/Get current device
  - Device synchronization
  - Device reset capabilities

- Memory Management
  - CUDA memory allocation
  - Memory copying (host-to-device, device-to-host, device-to-device)
  - Automatic resource cleanup
  - Unified memory support
  - Pinned memory operations
  - Memory pool with fragmentation handling

### cuBLAS Support
- High-performance matrix operations
- GEMM (General Matrix Multiplication)
- Optimized linear algebra operations
- Automatic handle management
- Batch processing capabilities

### cuDNN Support
- Deep learning primitives
- Convolution operations
  - Forward convolution
  - Backward convolution (data)
  - Backward convolution (filter)
- Pooling operations
  - Forward pooling
  - Backward pooling
- Activation functions
  - Forward activation
  - Backward activation

### Tensor Operations
- Tensor creation and manipulation
- Basic operations (add, multiply)
- Activation functions (ReLU, sigmoid, tanh)
- Gradient computation
- Shape manipulation

### Multi-GPU Support
- Device affinity management
- Load balancing
- Multi-GPU computation
- Device synchronization
- Thread safety

### Profiling and Monitoring
- CUDA event timing
- Memory usage tracking
- Kernel metrics collection
- Device utilization monitoring
- Performance benchmarking

### Error Handling
- Comprehensive error checking
- Error status retrieval
- Error message translation

## Requirements

- PHP 7.0 or later
- CUDA Toolkit 8.0 or later
- cuBLAS (included with CUDA Toolkit)
- cuDNN 7.0 or later
- C compiler (gcc/clang)
- PHP development files
- NVTX (optional, for profiling)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/php-cuda.git
cd php-cuda
```

2. Run the compile script:
```bash
./compile.sh
```

## Testing

The extension includes a comprehensive test suite covering various aspects of functionality:

### Running Tests

To run all tests after installation:
```bash
./compile.sh --test
```

To run individual test files:
```bash
cd tests/
php run-test.php test_name.phpt
```

### Available Tests

1. Basic Functionality (`001-basic.phpt`)
   - Device management
   - Basic matrix operations
   - Error handling

2. Memory Management (`002-memory.phpt`)
   - Memory allocation/deallocation
   - Memory pool operations
   - Leak detection

3. Stress Testing (`003-stress.phpt`)
   - High-load operations
   - Error recovery
   - Concurrent operations

4. Neural Network (`004-neural.phpt`)
   - Layer operations
   - Training functions
   - Model persistence

5. Tensor Operations (`005-tensor.phpt`)
   - Tensor creation
   - Basic operations
   - Activation functions
   - Gradient computation

6. Advanced Memory (`006-advanced-memory.phpt`)
   - Unified memory
   - Pinned memory
   - Memory pool fragmentation
   - Bandwidth measurement

7. Multi-GPU (`007-multi-gpu.phpt`)
   - Device management
   - Multi-GPU computation
   - Device synchronization
   - Thread safety

8. cuBLAS (`008-cublas.phpt`)
   - Basic operations
   - GEMM operations
   - Batch processing
   - Performance benchmarks

9. Profiling (`009-profiler.phpt`)
   - Event timing
   - Memory tracking
   - Kernel metrics
   - Device utilization

### Test Requirements

- Some tests require multiple GPUs (`007-multi-gpu.phpt`)
- Profiling tests require NVTX support (`009-profiler.phpt`)
- Memory tests require sufficient GPU memory
- Neural network tests require cuDNN

[Rest of the README content remains unchanged...]

## Usage Examples

[Previous usage examples remain unchanged...]

## API Reference

[Previous API reference remains unchanged...]

## Performance Considerations

[Previous performance considerations remain unchanged...]

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please use the GitHub issue tracker.
