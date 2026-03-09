#!/bin/bash

# Ensure the script stops on errors
set -e

# Bazel coverage produces empty reports with Clang on macOS. Use manual instrumentation instead.

echo "Building tests with instrumentation..."
bazel build \
    --compilation_mode=dbg \
    --spawn_strategy=local \
    --copt="-fprofile-instr-generate" \
    --copt="-fcoverage-mapping" \
    --linkopt="-fprofile-instr-generate" \
    //tests:splitnewton_coverage_tests

echo "Running tests..."
rm -f default.profraw
export LLVM_PROFILE_FILE="$(pwd)/default.profraw"
./bazel-bin/tests/splitnewton_coverage_tests

echo "Merging profile data..."
llvm-profdata merge -sparse default.profraw -o coverage.profdata

echo "Exporting to lcov format..."
EXEC_ROOT=$(bazel info execution_root)
WORKSPACE_ROOT=$(pwd)

llvm-cov export ./bazel-bin/tests/splitnewton_coverage_tests \
    -instr-profile=coverage.profdata \
    -format=lcov \
    -path-equivalence="$EXEC_ROOT","$WORKSPACE_ROOT" \
    -ignore-filename-regex='external/' \
    > coverage.lcov

echo "Generating HTML report..."
genhtml coverage.lcov --output-dir coverage_report --ignore-errors inconsistent

echo "Coverage report generated in ./coverage_report/index.html"
