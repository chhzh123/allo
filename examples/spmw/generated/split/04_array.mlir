#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @feed_up_load(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    %c0_i8 = arith.constant 0 : i8
    affine.for %arg3 = 0 to 4 {
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      affine.for %arg4 = 0 to 4 {
        affine.store %c0_i8, %alloc[%arg4] : memref<4xi8>
      }
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %arg0[%arg3, %arg4] {from = "local_At"} : memref<4x4xi8, #map>
        affine.store %0, %alloc[%arg4] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4xi8>, 16> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @feed_3_up_load(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    %c0_i8 = arith.constant 0 : i8
    affine.for %arg3 = 0 to 4 {
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      affine.for %arg4 = 0 to 4 {
        affine.store %c0_i8, %alloc[%arg4] : memref<4xi8>
      }
      affine.for %arg4 = 0 to 4 {
        %0 = affine.load %arg0[%arg3, %arg4] {from = "local_Bt"} : memref<4x4xi8, #map>
        affine.store %0, %alloc[%arg4] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4xi8>, 16> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r3(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r4(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r5(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r6(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r7(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r8(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_1[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @drain_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %0 = allo.stream_get(%arg3, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to %arg0 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @drain_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %0 = allo.stream_get(%arg3, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg4 = 0 to %arg0 {
      allo.stream_put(%arg2, [], %c0_i32) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @drain_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 4>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %0 = allo.stream_get(%arg3, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 4> contains i32
    affine.for %arg5 = 0 to %arg0 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 4> contains i32
    } {loop_name = "_i", op_name = "S__i_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_r0(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_r1(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 16> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_r2(%arg0: index, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_3_r0(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_3_r1(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 16> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_3_r2(%arg0: index, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @drain_down_drain(%arg0: memref<4x4xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 4> -> i32
      affine.store %0, %arg0[%arg1, %arg3] {to = "local_Ct"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<4x4xi8, #map>, %arg1: memref<4x4xi8, #map>, %arg2: memref<4x4xi32, #map>) attributes {dataflow, itypes = "sss", otypes = "", top} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %0 = allo.stream_construct() {name = "feed_3_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %1 = allo.stream_construct() {name = "feed_3_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %2 = allo.stream_construct() {name = "feed_3_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %3 = allo.stream_construct() {name = "feed_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %4 = allo.stream_construct() {name = "feed_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %5 = allo.stream_construct() {name = "feed_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %6 = allo.stream_construct() {name = "drain_down_bind_3"} : !allo.stream<i32, 4>
    %7 = allo.stream_construct() {name = "drain_down_bind_2"} : !allo.stream<i32, 4>
    %8 = allo.stream_construct() {name = "drain_down_bind_1"} : !allo.stream<i32, 4>
    %9 = allo.stream_construct() {name = "drain_down_bind_0"} : !allo.stream<i32, 4>
    %10 = allo.stream_construct() {name = "drain_down_up_3_3"} : !allo.stream<i32, 2>
    %11 = allo.stream_construct() {name = "drain_down_up_3_2"} : !allo.stream<i32, 2>
    %12 = allo.stream_construct() {name = "drain_down_up_3_1"} : !allo.stream<i32, 2>
    %13 = allo.stream_construct() {name = "drain_down_up_3_0"} : !allo.stream<i32, 2>
    %14 = allo.stream_construct() {name = "drain_down_up_2_3"} : !allo.stream<i32, 2>
    %15 = allo.stream_construct() {name = "drain_down_up_2_2"} : !allo.stream<i32, 2>
    %16 = allo.stream_construct() {name = "drain_down_up_2_1"} : !allo.stream<i32, 2>
    %17 = allo.stream_construct() {name = "drain_down_up_2_0"} : !allo.stream<i32, 2>
    %18 = allo.stream_construct() {name = "drain_down_up_1_3"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "drain_down_up_1_2"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "drain_down_up_1_1"} : !allo.stream<i32, 2>
    %21 = allo.stream_construct() {name = "drain_down_up_1_0"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "drain_mine_bind_15"} : !allo.stream<i32, 2>
    %23 = allo.stream_construct() {name = "pe_east_west_3_3"} : !allo.stream<i8, 2>
    %24 = allo.stream_construct() {name = "drain_mine_bind_14"} : !allo.stream<i32, 2>
    %25 = allo.stream_construct() {name = "pe_east_west_3_2"} : !allo.stream<i8, 2>
    %26 = allo.stream_construct() {name = "drain_mine_bind_13"} : !allo.stream<i32, 2>
    %27 = allo.stream_construct() {name = "pe_west_bind_3"} : !allo.stream<i8, 2>
    %28 = allo.stream_construct() {name = "pe_east_west_3_1"} : !allo.stream<i8, 2>
    %29 = allo.stream_construct() {name = "drain_mine_bind_12"} : !allo.stream<i32, 2>
    %30 = allo.stream_construct() {name = "pe_south_north_3_3"} : !allo.stream<i8, 2>
    %31 = allo.stream_construct() {name = "drain_mine_bind_11"} : !allo.stream<i32, 2>
    %32 = allo.stream_construct() {name = "pe_south_north_3_2"} : !allo.stream<i8, 2>
    %33 = allo.stream_construct() {name = "pe_east_west_2_3"} : !allo.stream<i8, 2>
    %34 = allo.stream_construct() {name = "drain_mine_bind_10"} : !allo.stream<i32, 2>
    %35 = allo.stream_construct() {name = "pe_south_north_3_1"} : !allo.stream<i8, 2>
    %36 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<i8, 2>
    %37 = allo.stream_construct() {name = "drain_mine_bind_9"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 2>
    %39 = allo.stream_construct() {name = "pe_south_north_3_0"} : !allo.stream<i8, 2>
    %40 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<i8, 2>
    %41 = allo.stream_construct() {name = "drain_mine_bind_8"} : !allo.stream<i32, 2>
    %42 = allo.stream_construct() {name = "pe_south_north_2_3"} : !allo.stream<i8, 2>
    %43 = allo.stream_construct() {name = "drain_mine_bind_7"} : !allo.stream<i32, 2>
    %44 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<i8, 2>
    %45 = allo.stream_construct() {name = "pe_east_west_1_3"} : !allo.stream<i8, 2>
    %46 = allo.stream_construct() {name = "drain_mine_bind_6"} : !allo.stream<i32, 2>
    %47 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<i8, 2>
    %48 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<i8, 2>
    %49 = allo.stream_construct() {name = "drain_mine_bind_5"} : !allo.stream<i32, 2>
    %50 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 2>
    %51 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<i8, 2>
    %52 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<i8, 2>
    %53 = allo.stream_construct() {name = "drain_mine_bind_4"} : !allo.stream<i32, 2>
    %54 = allo.stream_construct() {name = "pe_south_north_1_3"} : !allo.stream<i8, 2>
    %55 = allo.stream_construct() {name = "pe_north_bind_3"} : !allo.stream<i8, 2>
    %56 = allo.stream_construct() {name = "drain_mine_bind_3"} : !allo.stream<i32, 2>
    %57 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<i8, 2>
    %58 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 2>
    %59 = allo.stream_construct() {name = "pe_east_west_0_3"} : !allo.stream<i8, 2>
    %60 = allo.stream_construct() {name = "drain_mine_bind_2"} : !allo.stream<i32, 2>
    %61 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<i8, 2>
    %62 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 2>
    %63 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<i8, 2>
    %64 = allo.stream_construct() {name = "drain_mine_bind_1"} : !allo.stream<i32, 2>
    %65 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 2>
    %66 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<i8, 2>
    %67 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 2>
    %68 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<i8, 2>
    %69 = allo.stream_construct() {name = "drain_mine_bind_0"} : !allo.stream<i32, 2>
    %70 = allo.stream_construct() {name = "feed_3_up_bind_0"} : !allo.stream<memref<4xi8>, 16>
    %71 = allo.stream_construct() {name = "feed_up_bind_0"} : !allo.stream<memref<4xi8>, 16>
    call @feed_up_load(%arg0, %c0, %71) : (memref<4x4xi8, #map>, index, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_3_up_load(%arg1, %c0, %70) : (memref<4x4xi8, #map>, index, !allo.stream<memref<4xi8>, 16>) -> ()
    call @pe_r6(%c0, %c0, %69, %68, %67, %66, %65) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r3(%c0, %c1, %64, %63, %62, %61, %68) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r3(%c0, %c2, %60, %59, %58, %57, %63) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r8(%c0, %c3, %56, %55, %54, %59) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r1(%c1, %c0, %53, %52, %66, %51, %50) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c1, %c1, %49, %48, %61, %47, %52) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c1, %c2, %46, %45, %57, %44, %48) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r4(%c1, %c3, %43, %54, %42, %45) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r1(%c2, %c0, %41, %40, %51, %39, %38) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c2, %c1, %37, %36, %47, %35, %40) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c2, %c2, %34, %33, %44, %32, %36) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r4(%c2, %c3, %31, %42, %30, %33) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r5(%c3, %c0, %29, %28, %39, %27) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r2(%c3, %c1, %26, %25, %35, %28) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r2(%c3, %c2, %24, %23, %32, %25) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r7(%c3, %c3, %22, %30, %23) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @drain_r1(%c0, %c0, %21, %69) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r1(%c0, %c1, %20, %64) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r1(%c0, %c2, %19, %60) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r1(%c0, %c3, %18, %56) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c1, %c0, %17, %53, %21) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c1, %c1, %16, %49, %20) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c1, %c2, %15, %46, %19) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c1, %c3, %14, %43, %18) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c2, %c0, %13, %41, %17) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c2, %c1, %12, %37, %16) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c2, %c2, %11, %34, %15) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r0(%c2, %c3, %10, %31, %14) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r2(%c3, %c0, %9, %29, %13) : (index, index, !allo.stream<i32, 4>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r2(%c3, %c1, %8, %26, %12) : (index, index, !allo.stream<i32, 4>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r2(%c3, %c2, %7, %24, %11) : (index, index, !allo.stream<i32, 4>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @drain_r2(%c3, %c3, %6, %22, %10) : (index, index, !allo.stream<i32, 4>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @feed_r1(%c0, %5, %65, %71) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_r0(%c1, %4, %50, %5) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_r0(%c2, %3, %38, %4) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_r2(%c3, %27, %3) : (index, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_3_r1(%c0, %2, %67, %70) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_3_r0(%c1, %1, %62, %2) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_3_r0(%c2, %0, %58, %1) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_3_r2(%c3, %55, %0) : (index, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @drain_down_drain(%arg2, %c0, %9) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @drain_down_drain(%arg2, %c1, %8) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @drain_down_drain(%arg2, %c2, %7) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @drain_down_drain(%arg2, %c3, %6) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    return
  }
}
