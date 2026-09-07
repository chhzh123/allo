#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, 0, 0, d1)>
#map2 = affine_map<(d0, d1) -> (0, d1, d0, 0)>
module {
  func.func @PE_kernel_gemm_0_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_0(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_1(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_2(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_3(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_4(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_5(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_6(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_0_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_1_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_2_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_3_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_4_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_5_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_6_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @PE_kernel_gemm_7_7(%arg0: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg1: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg2: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg3: memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, %arg4: memref<8x8xi32, #map>, %arg5: index, %arg6: index) attributes {itypes = "sssss__", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32_0"} 0 : i32
    %alloc = memref.alloc() {name = "v"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "v"} : memref<i32>
    affine.for %arg7 = 0 to 8 {
      %1 = affine.load %arg0[%arg7] {from = "A_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = affine.load %arg1[%arg7] {from = "B_in"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "v"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "v"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      affine.store %13, %arg2[%arg7] {to = "A_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      affine.store %14, %arg3[%arg7] {to = "B_out"} : memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    } {loop_name = "k", op_name = "reduction", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "v"} : memref<i32>
    affine.store %0, %arg4[%arg5, %arg6] {to = "C"} : memref<8x8xi32, #map>
    return
  }
  func.func @systolic_tile_gemm(%arg0: memref<8x8xi8, #map1>, %arg1: memref<8x8xi8, #map2>, %arg2: memref<8x8xi32, #map>) attributes {dataflow, itypes = "sss", otypes = ""} {
    %alloc = memref.alloc() {name = "A_fifo"} : memref<8x9x8xi8, "stream:9;SST">
    %alloc_0 = memref.alloc() {name = "B_fifo"} : memref<8x9x8xi8, "stream:9;SST">
    %alloc_1 = memref.alloc() {name = "A_drain"} : memref<8xi8>
    %alloc_2 = memref.alloc() {name = "B_drain"} : memref<8xi8>
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 8 {
        %25 = affine.load %arg0[%arg4, %arg3] {from = "A"} : memref<8x8xi8, #map1>
        affine.store %25, %alloc[%arg4, 0, %arg3] {to = "A_fifo"} : memref<8x9x8xi8, "stream:9;SST">
      } {loop_name = "m", op_name = "S_m_0"}
      affine.for %arg4 = 0 to 8 {
        %25 = affine.load %arg1[%arg3, %arg4] {from = "B"} : memref<8x8xi8, #map2>
        affine.store %25, %alloc_0[%arg4, 0, %arg3] {to = "B_fifo"} : memref<8x9x8xi8, "stream:9;SST">
      } {loop_name = "n", op_name = "S_n_1"}
    } {loop_name = "k", op_name = "data_load"}
    %c0 = arith.constant 0 : index
    %subview = memref.subview %alloc[%c0, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_3 = memref.subview %alloc_0[%c0, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %0 = arith.index_cast %c0 : index to i34
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.extsi %c1_i32 : i32 to i34
    %2 = arith.addi %0, %1 : i34
    %3 = arith.index_cast %2 : i34 to index
    %subview_4 = memref.subview %alloc[%c0, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_5 = memref.subview %alloc_0[%c0, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_0(%subview, %subview_3, %subview_4, %subview_5, %arg2, %c0, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c1 = arith.constant 1 : index
    %subview_6 = memref.subview %alloc[%c0, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_7 = memref.subview %alloc_0[%c1, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %4 = arith.index_cast %c1 : index to i34
    %5 = arith.addi %4, %1 : i34
    %6 = arith.index_cast %5 : i34 to index
    %subview_8 = memref.subview %alloc[%c0, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_9 = memref.subview %alloc_0[%c1, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_0(%subview_6, %subview_7, %subview_8, %subview_9, %arg2, %c0, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c2 = arith.constant 2 : index
    %subview_10 = memref.subview %alloc[%c0, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_11 = memref.subview %alloc_0[%c2, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %7 = arith.index_cast %c2 : index to i34
    %8 = arith.addi %7, %1 : i34
    %9 = arith.index_cast %8 : i34 to index
    %subview_12 = memref.subview %alloc[%c0, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_13 = memref.subview %alloc_0[%c2, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_0(%subview_10, %subview_11, %subview_12, %subview_13, %arg2, %c0, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c3 = arith.constant 3 : index
    %subview_14 = memref.subview %alloc[%c0, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_15 = memref.subview %alloc_0[%c3, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %10 = arith.index_cast %c3 : index to i34
    %11 = arith.addi %10, %1 : i34
    %12 = arith.index_cast %11 : i34 to index
    %subview_16 = memref.subview %alloc[%c0, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_17 = memref.subview %alloc_0[%c3, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_0(%subview_14, %subview_15, %subview_16, %subview_17, %arg2, %c0, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c4 = arith.constant 4 : index
    %subview_18 = memref.subview %alloc[%c0, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_19 = memref.subview %alloc_0[%c4, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %13 = arith.index_cast %c4 : index to i34
    %14 = arith.addi %13, %1 : i34
    %15 = arith.index_cast %14 : i34 to index
    %subview_20 = memref.subview %alloc[%c0, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_21 = memref.subview %alloc_0[%c4, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_0(%subview_18, %subview_19, %subview_20, %subview_21, %arg2, %c0, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c5 = arith.constant 5 : index
    %subview_22 = memref.subview %alloc[%c0, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_23 = memref.subview %alloc_0[%c5, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %16 = arith.index_cast %c5 : index to i34
    %17 = arith.addi %16, %1 : i34
    %18 = arith.index_cast %17 : i34 to index
    %subview_24 = memref.subview %alloc[%c0, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_25 = memref.subview %alloc_0[%c5, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_0(%subview_22, %subview_23, %subview_24, %subview_25, %arg2, %c0, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c6 = arith.constant 6 : index
    %subview_26 = memref.subview %alloc[%c0, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_27 = memref.subview %alloc_0[%c6, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %19 = arith.index_cast %c6 : index to i34
    %20 = arith.addi %19, %1 : i34
    %21 = arith.index_cast %20 : i34 to index
    %subview_28 = memref.subview %alloc[%c0, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_29 = memref.subview %alloc_0[%c6, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_0(%subview_26, %subview_27, %subview_28, %subview_29, %arg2, %c0, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %c7 = arith.constant 7 : index
    %subview_30 = memref.subview %alloc[%c0, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_31 = memref.subview %alloc_0[%c7, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %22 = arith.index_cast %c7 : index to i34
    %23 = arith.addi %22, %1 : i34
    %24 = arith.index_cast %23 : i34 to index
    %subview_32 = memref.subview %alloc[%c0, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_33 = memref.subview %alloc_0[%c7, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_0(%subview_30, %subview_31, %subview_32, %subview_33, %arg2, %c0, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_34 = memref.subview %alloc[%c1, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_35 = memref.subview %alloc_0[%c0, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_36 = memref.subview %alloc[%c1, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_37 = memref.subview %alloc_0[%c0, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_1(%subview_34, %subview_35, %subview_36, %subview_37, %arg2, %c1, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_38 = memref.subview %alloc[%c1, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_39 = memref.subview %alloc_0[%c1, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_40 = memref.subview %alloc[%c1, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_41 = memref.subview %alloc_0[%c1, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_1(%subview_38, %subview_39, %subview_40, %subview_41, %arg2, %c1, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_42 = memref.subview %alloc[%c1, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_43 = memref.subview %alloc_0[%c2, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_44 = memref.subview %alloc[%c1, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_45 = memref.subview %alloc_0[%c2, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_1(%subview_42, %subview_43, %subview_44, %subview_45, %arg2, %c1, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_46 = memref.subview %alloc[%c1, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_47 = memref.subview %alloc_0[%c3, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_48 = memref.subview %alloc[%c1, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_49 = memref.subview %alloc_0[%c3, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_1(%subview_46, %subview_47, %subview_48, %subview_49, %arg2, %c1, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_50 = memref.subview %alloc[%c1, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_51 = memref.subview %alloc_0[%c4, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_52 = memref.subview %alloc[%c1, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_53 = memref.subview %alloc_0[%c4, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_1(%subview_50, %subview_51, %subview_52, %subview_53, %arg2, %c1, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_54 = memref.subview %alloc[%c1, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_55 = memref.subview %alloc_0[%c5, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_56 = memref.subview %alloc[%c1, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_57 = memref.subview %alloc_0[%c5, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_1(%subview_54, %subview_55, %subview_56, %subview_57, %arg2, %c1, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_58 = memref.subview %alloc[%c1, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_59 = memref.subview %alloc_0[%c6, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_60 = memref.subview %alloc[%c1, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_61 = memref.subview %alloc_0[%c6, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_1(%subview_58, %subview_59, %subview_60, %subview_61, %arg2, %c1, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_62 = memref.subview %alloc[%c1, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_63 = memref.subview %alloc_0[%c7, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_64 = memref.subview %alloc[%c1, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_65 = memref.subview %alloc_0[%c7, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_1(%subview_62, %subview_63, %subview_64, %subview_65, %arg2, %c1, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_66 = memref.subview %alloc[%c2, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_67 = memref.subview %alloc_0[%c0, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_68 = memref.subview %alloc[%c2, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_69 = memref.subview %alloc_0[%c0, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_2(%subview_66, %subview_67, %subview_68, %subview_69, %arg2, %c2, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_70 = memref.subview %alloc[%c2, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_71 = memref.subview %alloc_0[%c1, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_72 = memref.subview %alloc[%c2, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_73 = memref.subview %alloc_0[%c1, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_2(%subview_70, %subview_71, %subview_72, %subview_73, %arg2, %c2, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_74 = memref.subview %alloc[%c2, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_75 = memref.subview %alloc_0[%c2, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_76 = memref.subview %alloc[%c2, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_77 = memref.subview %alloc_0[%c2, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_2(%subview_74, %subview_75, %subview_76, %subview_77, %arg2, %c2, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_78 = memref.subview %alloc[%c2, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_79 = memref.subview %alloc_0[%c3, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_80 = memref.subview %alloc[%c2, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_81 = memref.subview %alloc_0[%c3, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_2(%subview_78, %subview_79, %subview_80, %subview_81, %arg2, %c2, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_82 = memref.subview %alloc[%c2, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_83 = memref.subview %alloc_0[%c4, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_84 = memref.subview %alloc[%c2, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_85 = memref.subview %alloc_0[%c4, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_2(%subview_82, %subview_83, %subview_84, %subview_85, %arg2, %c2, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_86 = memref.subview %alloc[%c2, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_87 = memref.subview %alloc_0[%c5, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_88 = memref.subview %alloc[%c2, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_89 = memref.subview %alloc_0[%c5, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_2(%subview_86, %subview_87, %subview_88, %subview_89, %arg2, %c2, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_90 = memref.subview %alloc[%c2, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_91 = memref.subview %alloc_0[%c6, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_92 = memref.subview %alloc[%c2, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_93 = memref.subview %alloc_0[%c6, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_2(%subview_90, %subview_91, %subview_92, %subview_93, %arg2, %c2, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_94 = memref.subview %alloc[%c2, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_95 = memref.subview %alloc_0[%c7, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_96 = memref.subview %alloc[%c2, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_97 = memref.subview %alloc_0[%c7, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_2(%subview_94, %subview_95, %subview_96, %subview_97, %arg2, %c2, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_98 = memref.subview %alloc[%c3, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_99 = memref.subview %alloc_0[%c0, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_100 = memref.subview %alloc[%c3, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_101 = memref.subview %alloc_0[%c0, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_3(%subview_98, %subview_99, %subview_100, %subview_101, %arg2, %c3, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_102 = memref.subview %alloc[%c3, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_103 = memref.subview %alloc_0[%c1, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_104 = memref.subview %alloc[%c3, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_105 = memref.subview %alloc_0[%c1, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_3(%subview_102, %subview_103, %subview_104, %subview_105, %arg2, %c3, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_106 = memref.subview %alloc[%c3, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_107 = memref.subview %alloc_0[%c2, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_108 = memref.subview %alloc[%c3, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_109 = memref.subview %alloc_0[%c2, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_3(%subview_106, %subview_107, %subview_108, %subview_109, %arg2, %c3, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_110 = memref.subview %alloc[%c3, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_111 = memref.subview %alloc_0[%c3, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_112 = memref.subview %alloc[%c3, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_113 = memref.subview %alloc_0[%c3, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_3(%subview_110, %subview_111, %subview_112, %subview_113, %arg2, %c3, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_114 = memref.subview %alloc[%c3, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_115 = memref.subview %alloc_0[%c4, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_116 = memref.subview %alloc[%c3, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_117 = memref.subview %alloc_0[%c4, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_3(%subview_114, %subview_115, %subview_116, %subview_117, %arg2, %c3, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_118 = memref.subview %alloc[%c3, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_119 = memref.subview %alloc_0[%c5, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_120 = memref.subview %alloc[%c3, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_121 = memref.subview %alloc_0[%c5, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_3(%subview_118, %subview_119, %subview_120, %subview_121, %arg2, %c3, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_122 = memref.subview %alloc[%c3, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_123 = memref.subview %alloc_0[%c6, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_124 = memref.subview %alloc[%c3, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_125 = memref.subview %alloc_0[%c6, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_3(%subview_122, %subview_123, %subview_124, %subview_125, %arg2, %c3, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_126 = memref.subview %alloc[%c3, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_127 = memref.subview %alloc_0[%c7, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_128 = memref.subview %alloc[%c3, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_129 = memref.subview %alloc_0[%c7, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_3(%subview_126, %subview_127, %subview_128, %subview_129, %arg2, %c3, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_130 = memref.subview %alloc[%c4, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_131 = memref.subview %alloc_0[%c0, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_132 = memref.subview %alloc[%c4, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_133 = memref.subview %alloc_0[%c0, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_4(%subview_130, %subview_131, %subview_132, %subview_133, %arg2, %c4, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_134 = memref.subview %alloc[%c4, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_135 = memref.subview %alloc_0[%c1, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_136 = memref.subview %alloc[%c4, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_137 = memref.subview %alloc_0[%c1, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_4(%subview_134, %subview_135, %subview_136, %subview_137, %arg2, %c4, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_138 = memref.subview %alloc[%c4, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_139 = memref.subview %alloc_0[%c2, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_140 = memref.subview %alloc[%c4, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_141 = memref.subview %alloc_0[%c2, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_4(%subview_138, %subview_139, %subview_140, %subview_141, %arg2, %c4, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_142 = memref.subview %alloc[%c4, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_143 = memref.subview %alloc_0[%c3, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_144 = memref.subview %alloc[%c4, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_145 = memref.subview %alloc_0[%c3, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_4(%subview_142, %subview_143, %subview_144, %subview_145, %arg2, %c4, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_146 = memref.subview %alloc[%c4, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_147 = memref.subview %alloc_0[%c4, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_148 = memref.subview %alloc[%c4, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_149 = memref.subview %alloc_0[%c4, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_4(%subview_146, %subview_147, %subview_148, %subview_149, %arg2, %c4, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_150 = memref.subview %alloc[%c4, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_151 = memref.subview %alloc_0[%c5, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_152 = memref.subview %alloc[%c4, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_153 = memref.subview %alloc_0[%c5, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_4(%subview_150, %subview_151, %subview_152, %subview_153, %arg2, %c4, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_154 = memref.subview %alloc[%c4, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_155 = memref.subview %alloc_0[%c6, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_156 = memref.subview %alloc[%c4, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_157 = memref.subview %alloc_0[%c6, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_4(%subview_154, %subview_155, %subview_156, %subview_157, %arg2, %c4, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_158 = memref.subview %alloc[%c4, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_159 = memref.subview %alloc_0[%c7, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_160 = memref.subview %alloc[%c4, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_161 = memref.subview %alloc_0[%c7, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_4(%subview_158, %subview_159, %subview_160, %subview_161, %arg2, %c4, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_162 = memref.subview %alloc[%c5, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_163 = memref.subview %alloc_0[%c0, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_164 = memref.subview %alloc[%c5, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_165 = memref.subview %alloc_0[%c0, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_5(%subview_162, %subview_163, %subview_164, %subview_165, %arg2, %c5, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_166 = memref.subview %alloc[%c5, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_167 = memref.subview %alloc_0[%c1, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_168 = memref.subview %alloc[%c5, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_169 = memref.subview %alloc_0[%c1, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_5(%subview_166, %subview_167, %subview_168, %subview_169, %arg2, %c5, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_170 = memref.subview %alloc[%c5, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_171 = memref.subview %alloc_0[%c2, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_172 = memref.subview %alloc[%c5, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_173 = memref.subview %alloc_0[%c2, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_5(%subview_170, %subview_171, %subview_172, %subview_173, %arg2, %c5, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_174 = memref.subview %alloc[%c5, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_175 = memref.subview %alloc_0[%c3, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_176 = memref.subview %alloc[%c5, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_177 = memref.subview %alloc_0[%c3, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_5(%subview_174, %subview_175, %subview_176, %subview_177, %arg2, %c5, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_178 = memref.subview %alloc[%c5, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_179 = memref.subview %alloc_0[%c4, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_180 = memref.subview %alloc[%c5, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_181 = memref.subview %alloc_0[%c4, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_5(%subview_178, %subview_179, %subview_180, %subview_181, %arg2, %c5, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_182 = memref.subview %alloc[%c5, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_183 = memref.subview %alloc_0[%c5, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_184 = memref.subview %alloc[%c5, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_185 = memref.subview %alloc_0[%c5, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_5(%subview_182, %subview_183, %subview_184, %subview_185, %arg2, %c5, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_186 = memref.subview %alloc[%c5, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_187 = memref.subview %alloc_0[%c6, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_188 = memref.subview %alloc[%c5, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_189 = memref.subview %alloc_0[%c6, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_5(%subview_186, %subview_187, %subview_188, %subview_189, %arg2, %c5, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_190 = memref.subview %alloc[%c5, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_191 = memref.subview %alloc_0[%c7, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_192 = memref.subview %alloc[%c5, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_193 = memref.subview %alloc_0[%c7, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_5(%subview_190, %subview_191, %subview_192, %subview_193, %arg2, %c5, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_194 = memref.subview %alloc[%c6, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_195 = memref.subview %alloc_0[%c0, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_196 = memref.subview %alloc[%c6, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_197 = memref.subview %alloc_0[%c0, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_6(%subview_194, %subview_195, %subview_196, %subview_197, %arg2, %c6, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_198 = memref.subview %alloc[%c6, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_199 = memref.subview %alloc_0[%c1, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_200 = memref.subview %alloc[%c6, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_201 = memref.subview %alloc_0[%c1, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_6(%subview_198, %subview_199, %subview_200, %subview_201, %arg2, %c6, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_202 = memref.subview %alloc[%c6, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_203 = memref.subview %alloc_0[%c2, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_204 = memref.subview %alloc[%c6, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_205 = memref.subview %alloc_0[%c2, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_6(%subview_202, %subview_203, %subview_204, %subview_205, %arg2, %c6, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_206 = memref.subview %alloc[%c6, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_207 = memref.subview %alloc_0[%c3, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_208 = memref.subview %alloc[%c6, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_209 = memref.subview %alloc_0[%c3, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_6(%subview_206, %subview_207, %subview_208, %subview_209, %arg2, %c6, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_210 = memref.subview %alloc[%c6, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_211 = memref.subview %alloc_0[%c4, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_212 = memref.subview %alloc[%c6, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_213 = memref.subview %alloc_0[%c4, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_6(%subview_210, %subview_211, %subview_212, %subview_213, %arg2, %c6, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_214 = memref.subview %alloc[%c6, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_215 = memref.subview %alloc_0[%c5, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_216 = memref.subview %alloc[%c6, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_217 = memref.subview %alloc_0[%c5, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_6(%subview_214, %subview_215, %subview_216, %subview_217, %arg2, %c6, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_218 = memref.subview %alloc[%c6, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_219 = memref.subview %alloc_0[%c6, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_220 = memref.subview %alloc[%c6, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_221 = memref.subview %alloc_0[%c6, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_6(%subview_218, %subview_219, %subview_220, %subview_221, %arg2, %c6, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_222 = memref.subview %alloc[%c6, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_223 = memref.subview %alloc_0[%c7, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_224 = memref.subview %alloc[%c6, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_225 = memref.subview %alloc_0[%c7, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_6(%subview_222, %subview_223, %subview_224, %subview_225, %arg2, %c6, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_226 = memref.subview %alloc[%c7, %c0, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_227 = memref.subview %alloc_0[%c0, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_228 = memref.subview %alloc[%c7, %3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_229 = memref.subview %alloc_0[%c0, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_0_7(%subview_226, %subview_227, %subview_228, %subview_229, %arg2, %c7, %c0) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_230 = memref.subview %alloc[%c7, %c1, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_231 = memref.subview %alloc_0[%c1, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_232 = memref.subview %alloc[%c7, %6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_233 = memref.subview %alloc_0[%c1, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_1_7(%subview_230, %subview_231, %subview_232, %subview_233, %arg2, %c7, %c1) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_234 = memref.subview %alloc[%c7, %c2, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_235 = memref.subview %alloc_0[%c2, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_236 = memref.subview %alloc[%c7, %9, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_237 = memref.subview %alloc_0[%c2, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_2_7(%subview_234, %subview_235, %subview_236, %subview_237, %arg2, %c7, %c2) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_238 = memref.subview %alloc[%c7, %c3, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_239 = memref.subview %alloc_0[%c3, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_240 = memref.subview %alloc[%c7, %12, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_241 = memref.subview %alloc_0[%c3, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_3_7(%subview_238, %subview_239, %subview_240, %subview_241, %arg2, %c7, %c3) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_242 = memref.subview %alloc[%c7, %c4, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_243 = memref.subview %alloc_0[%c4, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_244 = memref.subview %alloc[%c7, %15, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_245 = memref.subview %alloc_0[%c4, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_4_7(%subview_242, %subview_243, %subview_244, %subview_245, %arg2, %c7, %c4) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_246 = memref.subview %alloc[%c7, %c5, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_247 = memref.subview %alloc_0[%c5, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_248 = memref.subview %alloc[%c7, %18, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_249 = memref.subview %alloc_0[%c5, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_5_7(%subview_246, %subview_247, %subview_248, %subview_249, %arg2, %c7, %c5) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_250 = memref.subview %alloc[%c7, %c6, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_251 = memref.subview %alloc_0[%c6, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_252 = memref.subview %alloc[%c7, %21, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_253 = memref.subview %alloc_0[%c6, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_6_7(%subview_250, %subview_251, %subview_252, %subview_253, %arg2, %c7, %c6) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    %subview_254 = memref.subview %alloc[%c7, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_255 = memref.subview %alloc_0[%c7, %c7, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_256 = memref.subview %alloc[%c7, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    %subview_257 = memref.subview %alloc_0[%c7, %24, 0] [1, 1, 8] [1, 1, 1] : memref<8x9x8xi8, "stream:9;SST"> to memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">
    call @PE_kernel_gemm_7_7(%subview_254, %subview_255, %subview_256, %subview_257, %arg2, %c7, %c7) : (memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8xi8, strided<[1], offset: ?>, "stream:9;SST">, memref<8x8xi32, #map>, index, index) -> ()
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 8 {
        %25 = affine.load %alloc[%arg4, 8, %arg3] {from = "A_fifo"} : memref<8x9x8xi8, "stream:9;SST">
        affine.store %25, %alloc_1[%arg4] {to = "A_drain"} : memref<8xi8>
      } {loop_name = "m", op_name = "S_m_4"}
      affine.for %arg4 = 0 to 8 {
        %25 = affine.load %alloc_0[%arg4, 8, %arg3] {from = "B_fifo"} : memref<8x9x8xi8, "stream:9;SST">
        affine.store %25, %alloc_2[%arg4] {to = "B_drain"} : memref<8xi8>
      } {loop_name = "n", op_name = "S_n_5"}
    } {loop_name = "k", op_name = "data_drain"}
    return
  }
  func.func @systolic_gemm(%arg0: memref<8x8xi8>, %arg1: memref<8x8xi8>, %arg2: memref<8x8xi32>) attributes {itypes = "sss", otypes = ""} {
    %alloc = memref.alloc() {name = "local_A"} : memref<8x8xi8, #map1>
    %alloc_0 = memref.alloc() {name = "local_B"} : memref<8x8xi8, #map2>
    %alloc_1 = memref.alloc() {name = "local_C"} : memref<8x8xi32, #map>
    affine.for %arg3 = 0 to 1 {
      %c0 = arith.constant 0 : index
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %0 = arith.index_cast %c0 : index to i33
          %c0_i32 = arith.constant 0 : i32
          %1 = arith.extsi %c0_i32 : i32 to i33
          %2 = arith.cmpi eq, %0, %1 : i33
          scf.if %2 {
            %3 = affine.load %arg0[%arg3 * 8 + %arg5, %arg4] {from = "A"} : memref<8x8xi8>
            affine.store %3, %alloc[%arg5, %arg4] {to = "local_A"} : memref<8x8xi8, #map1>
          }
        } {loop_name = "ai"}
      } {loop_name = "ak", op_name = "load_A_tile", pipeline_ii = 1 : ui32}
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %0 = affine.load %arg1[%arg4, %c0 * 8 + %arg5] {from = "B"} : memref<8x8xi8>
          affine.store %0, %alloc_0[%arg4, %arg5] {to = "local_B"} : memref<8x8xi8, #map2>
        } {loop_name = "bj"}
      } {loop_name = "bk", op_name = "load_B_tile", pipeline_ii = 1 : ui32}
      func.call @systolic_tile_gemm(%alloc, %alloc_0, %alloc_1) : (memref<8x8xi8, #map1>, memref<8x8xi8, #map2>, memref<8x8xi32, #map>) -> ()
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 8 {
          %0 = affine.load %alloc_1[%arg5, %arg4] {from = "local_C"} : memref<8x8xi32, #map>
          affine.store %0, %arg2[%arg3 * 8 + %arg5, %c0 * 8 + %arg4] {to = "C"} : memref<8x8xi32>
        } {loop_name = "si"}
      } {loop_name = "sj", op_name = "store_C_tile", pipeline_ii = 1 : ui32}
    } {loop_name = "mi_ni_fused", op_name = "outer_tile"}
    return
  }
  func.func @gemm(%arg0: memref<8x8xi8>, %arg1: memref<8x8xi8>, %arg2: memref<8x8xi32>) attributes {itypes = "sss", otypes = ""} {
    call @systolic_gemm(%arg0, %arg1, %arg2) : (memref<8x8xi8>, memref<8x8xi8>, memref<8x8xi32>) -> ()
    return
  }
}
