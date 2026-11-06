# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Intel oneAPI 2025.0 DPC++ with its FPGA support package, unpacked from Intel's
# RPMs into a user-owned prefix by intel_unpack.sh (no root, no system ICD
# files). Source it before `set -u`: oneAPI's vars.sh reads variables it has not
# set, and it resets the positional parameters of the shell that sources it.
export ONEAPI_ROOT=/scratch/hc676/intel/root/opt/intel/oneapi
. $ONEAPI_ROOT/compiler/2025.0/env/vars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=$ONEAPI_ROOT/redist/lib:$ONEAPI_ROOT/tbb/2022.0/lib/intel64/gcc4.8:$ONEAPI_ROOT/umf/0.9/lib:$ONEAPI_ROOT/tcm/1.2/lib:${LD_LIBRARY_PATH:-}
# Only this prefix OpenCL drivers: the machine /etc/OpenCL/vendors holds the
# Xilinx XRT one, which aborts a compile when XILINX_XRT is not set.
export OCL_ICD_VENDORS=/scratch/hc676/intel/vendors
export OCL_ICD_FILENAMES=$ONEAPI_ROOT/compiler/2025.0/lib/libintelocl_emu.so:$ONEAPI_ROOT/compiler/2025.0/lib/libintelocl.so
