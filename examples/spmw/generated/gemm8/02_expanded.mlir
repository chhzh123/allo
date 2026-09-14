module {
  func.func @pe_west_load(%arg0: memref<3x3xi8>, %arg1: index, %arg2: !allo.stream<i8, 3>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg1, %arg3] {from = "local_A"} : memref<3x3xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 3> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_north_load(%arg0: memref<3x3xi8>, %arg1: index, %arg2: !allo.stream<i8, 3>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_B"} : memref<3x3xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 3> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_r0(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 3>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 3> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r1(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 3>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 3> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r2(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r3(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r4(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 3>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 3>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 3> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 3> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r5(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 3>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<i8, 3> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r6(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r7(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 3 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @pe_r8(%arg0: memref<3x3xi32>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 3>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<i8, 3> -> i8
      %alloc_2 = memref.alloc() {name = "b"} : memref<i8>
      affine.store %2, %alloc_2[] {to = "b"} : memref<i8>
      %3 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      %4 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = affine.load %alloc[] {from = "acc"} : memref<i32>
      %9 = arith.extsi %8 : i32 to i33
      %10 = arith.extsi %7 : i16 to i33
      %11 = arith.addi %9, %10 : i33
      %12 = arith.trunci %11 : i33 to i32
      affine.store %12, %alloc[] {to = "acc"} : memref<i32>
      %13 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xi32>
    return
  }
  func.func @top(%arg0: memref<3x3xi8>, %arg1: memref<3x3xi8>, %arg2: memref<3x3xi32>) attributes {dataflow, itypes = "sss", otypes = ""} {
    %0 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<i8, 2>
    %1 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<i8, 2>
    %2 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<i8, 2>
    %3 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<i8, 2>
    %4 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<i8, 2>
    %5 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<i8, 2>
    %6 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<i8, 2>
    %7 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<i8, 2>
    %8 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<i8, 2>
    %9 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 3>
    %13 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 3>
    %14 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 3>
    %15 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 3>
    %c2 = arith.constant 2 : index
    %16 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 3>
    %c1 = arith.constant 1 : index
    %17 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 3>
    %c0 = arith.constant 0 : index
    call @pe_west_load(%arg0, %c0, %17) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_west_load(%arg0, %c1, %16) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_west_load(%arg0, %c2, %15) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_north_load(%arg1, %c0, %14) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_north_load(%arg1, %c1, %13) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_north_load(%arg1, %c2, %12) : (memref<3x3xi8>, index, !allo.stream<i8, 3>) -> ()
    call @pe_r4(%arg2, %c0, %c0, %11, %14, %10, %17) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 3>, !allo.stream<i8, 2>, !allo.stream<i8, 3>) -> ()
    call @pe_r5(%arg2, %c0, %c1, %9, %13, %8, %11) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 3>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r8(%arg2, %c0, %c2, %12, %7, %9) : (memref<3x3xi32>, index, index, !allo.stream<i8, 3>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r1(%arg2, %c1, %c0, %6, %10, %5, %16) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 3>) -> ()
    call @pe_r2(%arg2, %c1, %c1, %4, %8, %3, %6) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r6(%arg2, %c1, %c2, %7, %2, %4) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%arg2, %c2, %c0, %1, %5, %15) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 3>) -> ()
    call @pe_r3(%arg2, %c2, %c1, %0, %3, %1) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r7(%arg2, %c2, %c2, %2, %0) : (memref<3x3xi32>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    return
  }
}
