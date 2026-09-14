#map = affine_map<(d0) -> (d0)>
module {
  func.func @feed_up_load(%arg0: memref<4x4xi8>, %arg1: index, %arg2: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %0 = arith.trunci %c0_i32_0 : i32 to i8
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      linalg.fill ins(%0 : i8) outs(%alloc : memref<4xi8>)
      affine.for %arg4 = 0 to 4 {
        %1 = affine.load %arg0[%arg3, %arg4] {from = "local_At"} : memref<4x4xi8>
        affine.store %1, %alloc[%arg4] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0"}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4xi8>, 16> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @feed_2_up_load(%arg0: memref<4x4xi8>, %arg1: index, %arg2: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %0 = arith.trunci %c0_i32_0 : i32 to i8
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      linalg.fill ins(%0 : i8) outs(%alloc : memref<4xi8>)
      affine.for %arg4 = 0 to 4 {
        %1 = affine.load %arg0[%arg3, %arg4] {from = "local_Bt"} : memref<4x4xi8>
        affine.store %1, %alloc[%arg4] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0"}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4xi8>, 16> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>, %arg7: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "________", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg8 = 0 to 4 {
      %1 = allo.stream_get(%arg7, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg6, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg8 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>, %arg7: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "________", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg8 = 0 to 4 {
      %1 = allo.stream_get(%arg7, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg6, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg8 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
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
      %13 = affine.load %alloc_2[] {from = "b"} : memref<i8>
      allo.stream_put(%arg5, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg7 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r3(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 4> contains i32
    affine.for %arg7 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 4> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r4(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
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
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg7 = 0 to #map(%arg0) {
      %c0_i32_1 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      allo.stream_put(%arg2, [], %c0_i32_2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r5(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<i8, 2> -> i8
      %alloc_1 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_1[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg5, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 4> contains i32
    affine.for %arg7 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 4> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r6(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
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
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 4> contains i32
    affine.for %arg6 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 4> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r7(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg7 = 0 to 4 {
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
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg7 = 0 to #map(%arg0) {
      %c0_i32_1 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      allo.stream_put(%arg2, [], %c0_i32_2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @pe_r8(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
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
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to #map(%arg0) {
      %c0_i32_1 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      allo.stream_put(%arg2, [], %c0_i32_2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1"}
    return
  }
  func.func @feed_r0(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_r1(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 16> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_r2(%arg0: index, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_2_r0(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_2_r1(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 16> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_2_r2(%arg0: index, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @pe_c_out_drain(%arg0: memref<4x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 4> -> i32
      affine.store %0, %arg0[%arg1, %arg3] {to = "local_Ct"} : memref<4x4xi32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<4x4xi8>, %arg1: memref<4x4xi8>, %arg2: memref<4x4xi32>) attributes {dataflow, itypes = "sss", otypes = ""} {
    %0 = allo.stream_construct() {name = "feed_2_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %1 = allo.stream_construct() {name = "feed_2_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %2 = allo.stream_construct() {name = "feed_2_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %3 = allo.stream_construct() {name = "feed_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %4 = allo.stream_construct() {name = "feed_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %5 = allo.stream_construct() {name = "feed_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %6 = allo.stream_construct() {name = "pe_c_out_bind_3"} : !allo.stream<i32, 4>
    %7 = allo.stream_construct() {name = "pe_east_west_3_3"} : !allo.stream<i8, 2>
    %8 = allo.stream_construct() {name = "pe_c_out_bind_2"} : !allo.stream<i32, 4>
    %9 = allo.stream_construct() {name = "pe_east_west_3_2"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "pe_c_out_bind_1"} : !allo.stream<i32, 4>
    %11 = allo.stream_construct() {name = "pe_west_bind_3"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "pe_east_west_3_1"} : !allo.stream<i8, 2>
    %13 = allo.stream_construct() {name = "pe_c_out_bind_0"} : !allo.stream<i32, 4>
    %14 = allo.stream_construct() {name = "pe_south_north_3_3"} : !allo.stream<i8, 2>
    %15 = allo.stream_construct() {name = "pe_c_out_c_in_3_3"} : !allo.stream<i32, 2>
    %16 = allo.stream_construct() {name = "pe_south_north_3_2"} : !allo.stream<i8, 2>
    %17 = allo.stream_construct() {name = "pe_east_west_2_3"} : !allo.stream<i8, 2>
    %18 = allo.stream_construct() {name = "pe_c_out_c_in_3_2"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "pe_south_north_3_1"} : !allo.stream<i8, 2>
    %20 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "pe_c_out_c_in_3_1"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 2>
    %23 = allo.stream_construct() {name = "pe_south_north_3_0"} : !allo.stream<i8, 2>
    %24 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<i8, 2>
    %25 = allo.stream_construct() {name = "pe_c_out_c_in_3_0"} : !allo.stream<i32, 2>
    %26 = allo.stream_construct() {name = "pe_south_north_2_3"} : !allo.stream<i8, 2>
    %27 = allo.stream_construct() {name = "pe_c_out_c_in_2_3"} : !allo.stream<i32, 2>
    %28 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<i8, 2>
    %29 = allo.stream_construct() {name = "pe_east_west_1_3"} : !allo.stream<i8, 2>
    %30 = allo.stream_construct() {name = "pe_c_out_c_in_2_2"} : !allo.stream<i32, 2>
    %31 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<i8, 2>
    %32 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<i8, 2>
    %33 = allo.stream_construct() {name = "pe_c_out_c_in_2_1"} : !allo.stream<i32, 2>
    %34 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 2>
    %35 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<i8, 2>
    %36 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<i8, 2>
    %37 = allo.stream_construct() {name = "pe_c_out_c_in_2_0"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "pe_south_north_1_3"} : !allo.stream<i8, 2>
    %39 = allo.stream_construct() {name = "pe_north_bind_3"} : !allo.stream<i8, 2>
    %40 = allo.stream_construct() {name = "pe_c_out_c_in_1_3"} : !allo.stream<i32, 2>
    %c3 = arith.constant 3 : index
    %41 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<i8, 2>
    %42 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 2>
    %43 = allo.stream_construct() {name = "pe_east_west_0_3"} : !allo.stream<i8, 2>
    %44 = allo.stream_construct() {name = "pe_c_out_c_in_1_2"} : !allo.stream<i32, 2>
    %c2 = arith.constant 2 : index
    %45 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<i8, 2>
    %46 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 2>
    %47 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<i8, 2>
    %48 = allo.stream_construct() {name = "pe_c_out_c_in_1_1"} : !allo.stream<i32, 2>
    %c1 = arith.constant 1 : index
    %49 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 2>
    %50 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<i8, 2>
    %51 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 2>
    %52 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<i8, 2>
    %53 = allo.stream_construct() {name = "pe_c_out_c_in_1_0"} : !allo.stream<i32, 2>
    %54 = allo.stream_construct() {name = "feed_2_up_bind_0"} : !allo.stream<memref<4xi8>, 16>
    %55 = allo.stream_construct() {name = "feed_up_bind_0"} : !allo.stream<memref<4xi8>, 16>
    %c0 = arith.constant 0 : index
    call @feed_up_load(%arg0, %c0, %55) : (memref<4x4xi8>, index, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_2_up_load(%arg1, %c0, %54) : (memref<4x4xi8>, index, !allo.stream<memref<4xi8>, 16>) -> ()
    call @pe_r7(%c0, %c0, %53, %52, %51, %50, %49) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r4(%c0, %c1, %48, %47, %46, %45, %52) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r4(%c0, %c2, %44, %43, %42, %41, %47) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r8(%c0, %c3, %40, %39, %38, %43) : (index, index, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r1(%c1, %c0, %53, %37, %36, %50, %35, %34) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c1, %c1, %48, %33, %32, %45, %31, %36) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c1, %c2, %44, %30, %29, %41, %28, %32) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r2(%c1, %c3, %40, %27, %38, %26, %29) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r1(%c2, %c0, %37, %25, %24, %35, %23, %22) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c2, %c1, %33, %21, %20, %31, %19, %24) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r0(%c2, %c2, %30, %18, %17, %28, %16, %20) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r2(%c2, %c3, %27, %15, %26, %14, %17) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r5(%c3, %c0, %25, %13, %12, %23, %11) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r3(%c3, %c1, %21, %10, %9, %19, %12) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r3(%c3, %c2, %18, %8, %7, %16, %9) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_r6(%c3, %c3, %15, %6, %14, %7) : (index, index, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @feed_r1(%c0, %5, %49, %55) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_r0(%c1, %4, %34, %5) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_r0(%c2, %3, %22, %4) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_r2(%c3, %11, %3) : (index, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_r1(%c0, %2, %51, %54) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 16>) -> ()
    call @feed_2_r0(%c1, %1, %46, %2) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_r0(%c2, %0, %42, %1) : (index, !allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_r2(%c3, %39, %0) : (index, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @pe_c_out_drain(%arg2, %c0, %13) : (memref<4x4xi32>, index, !allo.stream<i32, 4>) -> ()
    call @pe_c_out_drain(%arg2, %c1, %10) : (memref<4x4xi32>, index, !allo.stream<i32, 4>) -> ()
    call @pe_c_out_drain(%arg2, %c2, %8) : (memref<4x4xi32>, index, !allo.stream<i32, 4>) -> ()
    call @pe_c_out_drain(%arg2, %c3, %6) : (memref<4x4xi32>, index, !allo.stream<i32, 4>) -> ()
    return
  }
}
