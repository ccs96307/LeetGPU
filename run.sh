#!/bin/bash
set -e  # Stop on first error

PROBLEM_PATH=$1
if [ -z "$PROBLEM_PATH" ]; then
    echo "Usage: $0 <path-to-problem-dir>"
    exit 1
fi

SRC_DIR="$PROBLEM_PATH"
BUILD_DIR="build/$PROBLEM_PATH"

# Create build directory
mkdir -p $BUILD_DIR

echo "--- Compiling $PROBLEM_PATH ---"

if [ -f "$SRC_DIR/main.cpp" ]; then
    # Step 1: Compile CUDA kernel(s) with nvcc
    for cu in $SRC_DIR/*.cu; do
        nvcc -std=c++17 -I./CUDA -c "$cu" -o "$BUILD_DIR/$(basename ${cu%.cu}).o"
    done

    # Step 2: Compile main.cpp with g++ and link CUDA objects
    g++ -std=c++20 -I./CUDA \
        $SRC_DIR/main.cpp $BUILD_DIR/*.o \
        -L/usr/local/cuda/lib64 -lcudart \
        -o $BUILD_DIR/main
else
    # Pure CUDA solution
    nvcc -std=c++17 -I./CUDA \
        $SRC_DIR/*.cu \
        -o $BUILD_DIR/main
fi

echo "--- Running $PROBLEM_PATH ---"
$BUILD_DIR/main
