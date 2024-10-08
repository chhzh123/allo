#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Slapo, Amazon.com, Inc. Apache-2.0
# https://github.com/awslabs/slapo/blob/main/scripts/lint/git-black.sh


set -e
set -u
set -o pipefail

if [[ "$1" == "-i" ]]; then
    INPLACE_FORMAT=1
    shift 1
else
    INPLACE_FORMAT=0
fi

if [[ "$#" -lt 1 ]]; then
    echo "Usage: tests/git-clang-format.sh [-i] <commit>"
    echo ""
    echo "Run clang-format on C/C++ files that changed since <commit>"
    echo "Examples:"
    echo "- Compare last one commit: tests/git-clang-format.sh HEAD~1"
    echo "- Compare against upstream/main: tests/git-clang-format.sh upstream/main"
    echo "The -i will use clang-format to format files in-place instead of checking them."
    exit 1
fi

# required to make clang-format's dep click to work
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Print out specific version
echo "Version Information: $(clang-format --version)"

# Compute C/C++ files which changed to compare.
IFS=$'\n' read -a FILES -d'\n' < <(git diff --name-only --diff-filter=ACMRTUX $1 -- "*.c" "*.cpp") || true
echo "Read returned $?"
if [ -z ${FILES+x} ]; then
    echo "No changes in C/C++ files"
    exit 0
fi
echo "Files: ${FILES[@]}"

if [[ ${INPLACE_FORMAT} -eq 1 ]]; then
    echo "Running clang-format on C/C++ files against revision" $1:
    CMD=( "clang-format" "-i" "${FILES[@]}" )
    echo "${CMD[@]}"
    "${CMD[@]}"
else
    echo "Running clang-format in checking mode"
    clang-format --dry-run --Werror ${FILES[@]}
fi