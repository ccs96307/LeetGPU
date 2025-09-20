#!/bin/bash
set -e

PROBLEM_PATH=$1
KERNEL_FILE=$2
if [ -z "$PROBLEM_PATH" ]; then
    echo "Usage: $0 <path-to-problem-dir> [kernel.cu]"
    exit 1
fi

SRC_DIR="$PROBLEM_PATH"
BUILD_DIR="build/$PROBLEM_PATH"

mkdir -p $BUILD_DIR

echo "--- Compiling $PROBLEM_PATH ---"

if [ -f "$SRC_DIR/main.cpp" ]; then
    if [ -n "$KERNEL_FILE" ]; then
        nvcc -std=c++17 -I./CUDA -c "$SRC_DIR/$KERNEL_FILE" -o "$BUILD_DIR/kernel.o"
    else
        for cu in $SRC_DIR/*.cu; do
            nvcc -std=c++17 -I./CUDA -c "$cu" -o "$BUILD_DIR/$(basename ${cu%.cu}).o"
        done
    fi

    g++ -std=c++20 -I./CUDA \
        $SRC_DIR/main.cpp $BUILD_DIR/*.o \
        -L/usr/local/cuda/lib64 -lcudart \
        -o $BUILD_DIR/main
else
    if [ -n "$KERNEL_FILE" ]; then
        nvcc -std=c++17 -I./CUDA "$SRC_DIR/$KERNEL_FILE" -o $BUILD_DIR/main
    else
        nvcc -std=c++17 -I./CUDA $SRC_DIR/*.cu -o $BUILD_DIR/main
    fi
fi

echo "--- Running $PROBLEM_PATH ---"
$BUILD_DIR/main
