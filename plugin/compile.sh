#!/bin/bash

# Exit on error
set -e

# Function to detect CUDA installation
find_cuda() {
    local cuda_paths=("/usr/local/cuda" "/usr/local/cuda-"* "/opt/cuda" "/usr/cuda")
    for path in "${cuda_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# Function to detect PHP development files
find_php_config() {
    local php_config_paths=("php-config" "/usr/local/bin/php-config" "/usr/bin/php-config")
    for path in "${php_config_paths[@]}"; do
        if command -v "$path" >/dev/null 2>&1; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# Function to check CUDA compatibility
check_cuda_compatibility() {
    local cuda_dir="$1"
    local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    local major_version=$(echo $cuda_version | cut -d. -f1)
    
    if [ "$major_version" -lt 8 ]; then
        echo "Error: CUDA 8.0 or higher is required (found $cuda_version)"
        exit 1
    fi
    
    # Check for compatible GPU
    local has_compatible_gpu=$(nvidia-smi --query-gpu=compute_cap_major --format=csv,noheader | awk '$1 >= 3 {print}')
    if [ -z "$has_compatible_gpu" ]; then
        echo "Error: No compatible GPU found (requires compute capability 3.0 or higher)"
        exit 1
    fi
}

# Function to check cuDNN
check_cudnn() {
    local cudnn_dir="$1"
    if [ ! -f "$cudnn_dir/include/cudnn.h" ]; then
        echo "Error: cuDNN not found in $cudnn_dir"
        exit 1
    fi
    
    local cudnn_version=$(grep CUDNN_MAJOR "$cudnn_dir/include/cudnn.h" | awk '{print $3}')
    if [ "$cudnn_version" -lt 7 ]; then
        echo "Error: cuDNN 7.0 or higher is required"
        exit 1
    fi
}

echo "Checking prerequisites..."

# Check for PHP development files
PHP_CONFIG=$(find_php_config)
if [ -z "$PHP_CONFIG" ]; then
    echo "Error: PHP development files not found. Please install PHP development package."
    exit 1
fi

# Get PHP extension directory
PHP_EXTENSION_DIR=$("$PHP_CONFIG" --extension-dir)
echo "PHP extension directory: $PHP_EXTENSION_DIR"

# Check for CUDA installation
CUDA_PATH=$(find_cuda)
if [ -z "$CUDA_PATH" ]; then
    echo "Error: CUDA installation not found."
    exit 1
fi
echo "CUDA installation found at: $CUDA_PATH"

# Check CUDA compatibility
check_cuda_compatibility "$CUDA_PATH"

# Check for cuDNN if specified
if [ -n "$CUDNN_PATH" ]; then
    check_cudnn "$CUDNN_PATH"
fi

# Clean previous build files
echo "Cleaning previous build files..."
rm -rf .libs modules *.lo *.la *.o config.* Makefile* build libtool
make clean-cuda 2>/dev/null || true

# Generate configure script
echo "Running phpize..."
phpize

# Configure the build
echo "Configuring build..."
CONFIGURE_ARGS="--with-cuda=$CUDA_PATH"
if [ -n "$CUDNN_PATH" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-cudnn=$CUDNN_PATH"
fi
if [ -n "$NVTX_PATH" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-nvtx=$NVTX_PATH"
fi
if [ "$ENABLE_OPENMP" = "1" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --enable-openmp"
fi

./configure $CONFIGURE_ARGS

# Build the extension
echo "Building extension..."
make clean
make

# Verify the build
echo "Verifying build..."
if [ ! -f "modules/cuda.so" ]; then
    echo "Error: Build failed - cuda.so not found"
    exit 1
fi

# Install the extension
echo "Installing extension..."
sudo make install

# Update PHP configuration
echo "Updating PHP configuration..."
PHP_INI_DIR=$("$PHP_CONFIG" --ini-dir)
if [ ! -f "$PHP_INI_DIR/cuda.ini" ]; then
    echo "extension=cuda.so" | sudo tee "$PHP_INI_DIR/cuda.ini"
fi

# Run tests if requested
if [ "$1" = "--test" ]; then
    echo "Running tests..."
    make test
fi

echo "Build complete!"
echo "Please restart your PHP server/process for the changes to take effect."

# Verify installation
php -m | grep -q "cuda"
if [ $? -eq 0 ]; then
    echo "CUDA extension successfully installed and enabled"
else
    echo "Warning: CUDA extension installed but not enabled in PHP"
    echo "Please check your PHP configuration"
fi
