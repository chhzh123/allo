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
    spmw.map(%arg0) topology = <grid = [3], families = [#spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "chan", family = "pe_west_bind", kind = "table", slots = dense<[0, 1, 2]> : tensor<3xi32>>]> roles = [#spmw.role<unit = @pe_west_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<3xi32> : memref<3x3xi8>
    spmw.map(%arg1) topology = <grid = [3], families = [#spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "chan", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2]> : tensor<3xi32>>]> roles = [#spmw.role<unit = @pe_north_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<3xi32> : memref<3x3xi8>
    spmw.map(%arg2) topology = <grid = [3, 3], families = [#spmw.family<name = "pe_east_west", type = i8, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_south_north", type = i8, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 3, shape = [3]>, #spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, -1, 1, -1, -1, 2, -1, -1]> : tensor<9xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, -1, -1, -1, -1, -1, -1]> : tensor<9xi32>>, #spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @pe_r0, missing = ["south", "west"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r1, missing = ["west"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r2, missing = [], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r3, missing = ["south"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r4, missing = ["north", "west"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r5, missing = ["north"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r6, missing = ["east"], ports = ["north", "south", "west"]>, #spmw.role<unit = @pe_r7, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_r8, missing = ["east", "north"], ports = ["north", "south", "west"]>] classes = dense<[4, 5, 8, 1, 2, 6, 0, 3, 7]> : tensor<9xi32> : memref<3x3xi32>
    return
  }
}
