#!/bin/bash

# Ensure the script stops on errors
set -e

# Step 1: Run the test executable
echo "Running tests..."
./bazel-bin/tests/splitnewton_tests

# Step 2: Merge raw profile data
echo "Merging profile data..."
llvm-profdata merge -sparse default.profraw -o coverage.profdata

# Step 3: Generate coverage report
echo "Generating coverage report..."
llvm-cov show ./bazel-bin/tests/splitnewton_tests \
    -instr-profile=coverage.profdata \
    -ignore-filename-regex='(.*third_party.*|.*external.*|.*tests.*)' \
    -format=html \
    -output-dir=coverage_report

# Notify the user of success
echo "Coverage report generated in ./coverage_report."