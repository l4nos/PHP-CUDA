dnl config.m4 for extension cuda

PHP_ARG_WITH([cuda],
  [for CUDA support],
  [AS_HELP_STRING([--with-cuda=DIR],
    [Include CUDA support. DIR is the CUDA installation directory])],
  [no],
  [no])

PHP_ARG_WITH([cudnn],
  [for cuDNN support],
  [AS_HELP_STRING([--with-cudnn=DIR],
    [Include cuDNN support. DIR is the cuDNN installation directory])],
  [no],
  [no])

PHP_ARG_WITH([nvtx],
  [for NVTX support],
  [AS_HELP_STRING([--with-nvtx=DIR],
    [Include NVTX support for profiling. DIR is the NVTX installation directory])],
  [no],
  [no])

PHP_ARG_ENABLE([openmp],
  [whether to enable OpenMP support],
  [AS_HELP_STRING([--enable-openmp],
    [Enable OpenMP support for CPU fallback])],
  [no],
  [no])

if test "$PHP_CUDA" != "no"; then
    dnl Check for CUDA installation
    if test "$PHP_CUDA" = "yes"; then
        AC_PATH_PROG(NVCC, nvcc, no)
        if test "$NVCC" = "no"; then
            AC_MSG_ERROR([Cannot find NVCC. Please specify CUDA installation directory])
        fi
        CUDA_DIR=`dirname "$NVCC"`
        CUDA_DIR=`dirname "$CUDA_DIR"`
    else
        CUDA_DIR=$PHP_CUDA
    fi

    AC_MSG_CHECKING([for CUDA installation])
    if test ! -f "$CUDA_DIR/include/cuda.h"; then
        AC_MSG_ERROR([CUDA headers not found])
    fi
    AC_MSG_RESULT([found])

    dnl Check CUDA version and compute capability
    AC_MSG_CHECKING([CUDA version and compute capability])
    cat > conftest.cu <<EOF
    #include <cuda_runtime.h>
    #include <stdio.h>
    int main() {
        int driver_version, runtime_version;
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        printf("%d %d\n", driver_version, runtime_version);
        
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("%d.%d\n", prop.major, prop.minor);
        }
        return 0;
    }
EOF

    if $NVCC conftest.cu -o conftest; then
        version_info=`./conftest`
        driver_version=`echo $version_info | cut -d' ' -f1`
        runtime_version=`echo $version_info | cut -d' ' -f2`
        compute_capability=`echo $version_info | cut -d' ' -f3`
        
        if test "$driver_version" -lt 8000; then
            AC_MSG_ERROR([CUDA 8.0 or higher required])
        fi
        
        major_version=`echo $compute_capability | cut -d'.' -f1`
        if test "$major_version" -lt 3; then
            AC_MSG_ERROR([GPU with compute capability 3.0 or higher required])
        fi
        
        AC_MSG_RESULT([driver: $driver_version, runtime: $runtime_version, compute: $compute_capability])
    else
        AC_MSG_ERROR([Failed to compile CUDA test program])
    fi
    
    rm -f conftest.cu conftest

    dnl Check for cuDNN if specified
    if test "$PHP_CUDNN" != "no"; then
        if test "$PHP_CUDNN" != "yes"; then
            CUDNN_DIR=$PHP_CUDNN
        else
            CUDNN_DIR=$CUDA_DIR
        fi

        AC_MSG_CHECKING([for cuDNN])
        if test -f "$CUDNN_DIR/include/cudnn.h"; then
            PHP_ADD_INCLUDE($CUDNN_DIR/include)
            PHP_ADD_LIBRARY_WITH_PATH(cudnn, $CUDNN_DIR/lib64)
            AC_DEFINE(HAVE_CUDNN, 1, [Whether you have cuDNN])
            AC_MSG_RESULT([found])
        else
            AC_MSG_ERROR([cuDNN not found])
        fi
    fi

    dnl Check for NVTX if specified
    if test "$PHP_NVTX" != "no"; then
        if test "$PHP_NVTX" != "yes"; then
            NVTX_DIR=$PHP_NVTX
        else
            NVTX_DIR=$CUDA_DIR
        fi

        AC_MSG_CHECKING([for NVTX])
        if test -f "$NVTX_DIR/include/nvToolsExt.h"; then
            PHP_ADD_INCLUDE($NVTX_DIR/include)
            PHP_ADD_LIBRARY_WITH_PATH(nvToolsExt, $NVTX_DIR/lib64)
            AC_DEFINE(HAVE_NVTX, 1, [Whether you have NVTX])
            AC_MSG_RESULT([found])
        else
            AC_MSG_ERROR([NVTX not found])
        fi
    fi

    dnl Check for OpenMP if enabled
    if test "$PHP_OPENMP" != "no"; then
        AC_MSG_CHECKING([for OpenMP support])
        AC_LANG_PUSH([C])
        
        ORIG_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -fopenmp"
        
        AC_TRY_LINK(
            [#include <omp.h>],
            [omp_get_num_threads();],
            [
                AC_MSG_RESULT([yes])
                AC_DEFINE(HAVE_OPENMP, 1, [Whether you have OpenMP])
                EXTRA_CFLAGS="$EXTRA_CFLAGS -fopenmp"
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -fopenmp"
            ],
            [
                AC_MSG_ERROR([OpenMP not available])
            ]
        )
        
        CFLAGS="$ORIG_CFLAGS"
        AC_LANG_POP([C])
    fi

    dnl Platform specific settings
    case $host_os in
        darwin*)
            CUDA_CFLAGS="-arch=sm_30"
            ;;
        linux*)
            CUDA_CFLAGS="-arch=sm_30"
            ;;
    esac

    PHP_ADD_INCLUDE($CUDA_DIR/include)
    PHP_ADD_LIBRARY_WITH_PATH(cudart, $CUDA_DIR/lib64)
    PHP_ADD_LIBRARY_WITH_PATH(cublas, $CUDA_DIR/lib64)

    PHP_SUBST(CUDA_CFLAGS)

    PHP_NEW_EXTENSION(cuda, cuda.c cuda_kernel.cu memory_pool.cu matrix_ops.cu conv_ops.cu cpu_ops.cu tensor_ops.cu neural_net.cu profiler.cu, $ext_shared)
fi
