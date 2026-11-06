#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# intel_unpack.sh <prefix>: Intel oneAPI 2025.0 DPC++ with its FPGA support
# package -- the last release that compiles SYCL for an FPGA; from 2025.1 the
# compiler rejects -fintelfpga -- from Intel's own package repository, without
# root: download the packages named in intel_packages.txt, check each against
# the SHA-256 the repository lists for it, and unpack them under <prefix>/root.
# Unpacking them accepts Intel's developer-tools license.
# The packages' install scripts are not run: they write /etc/OpenCL/vendors
# and /etc/ld.so.conf.d, which env.sh replaces with OCL_ICD_VENDORS and
# LD_LIBRARY_PATH.
main() {
  set -u
  local prefix=$1 here
  here=$(cd "$(dirname "$0")" && pwd)
  mkdir -p "$prefix/rpms" "$prefix/root" "$prefix/vendors"
  cd "$prefix/rpms" || exit 1
  while read -r _type sum file; do
    [ -f "$file" ] || curl -s --fail -o "$file" "https://yum.repos.intel.com/oneapi/$file" || exit 1
    echo "$sum  $file" | sha256sum -c --quiet - || exit 1
  done < "$here/intel_packages.txt"
  cd "$prefix/root" || exit 1
  for r in "$prefix"/rpms/*.rpm; do rpm2cpio "$r" | cpio -idm --quiet; done
  local o=$prefix/root/opt/intel/oneapi
  ln -sfT 2025.0 "$o/compiler/latest"
  echo "$o/compiler/2025.0/lib/libintelocl_emu.so" > "$prefix/vendors/Intel_FPGA_SSG_Emulator.icd"
  echo "$o/compiler/2025.0/lib/libintelocl.so" > "$prefix/vendors/intel64.icd"
}
main "$@"
