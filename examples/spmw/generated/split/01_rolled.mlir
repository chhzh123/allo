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
  func.func @feed_3_up_load(%arg0: memref<4x4xi8>, %arg1: index, %arg2: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "s__", otypes = ""} {
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
  func.func @pe_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
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
    return
  }
  func.func @pe_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
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
    return
  }
  func.func @pe_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
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
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r3(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
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
    return
  }
  func.func @pe_r4(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
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
    return
  }
  func.func @pe_r5(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
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
      %13 = affine.load %alloc_1[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    return
  }
  func.func @pe_r6(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
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
    return
  }
  func.func @pe_r7(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32_0, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
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
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
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
    return
  }
  func.func @drain_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_0"}
    return
  }
  func.func @drain_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg4 = 0 to #map(%arg0) {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      allo.stream_put(%arg2, [], %c0_i32_0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_0"}
    return
  }
  func.func @drain_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<i32, 4>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
    allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 4> contains i32
    affine.for %arg5 = 0 to #map(%arg0) {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 4> contains i32
    } {loop_name = "_i", op_name = "S__i_0"}
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
  func.func @feed_3_r0(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_3_r1(%arg0: index, %arg1: !allo.stream<memref<4xi8>, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<memref<4xi8>, 16>) attributes {df.kernel, itypes = "____", otypes = ""} {
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg3, []) {name = "packed"} : !allo.stream<memref<4xi8>, 16> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg1, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @feed_3_r2(%arg0: index, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[%arg0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0"}
    return
  }
  func.func @drain_down_drain(%arg0: memref<4x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 4> -> i32
      affine.store %0, %arg0[%arg1, %arg3] {to = "local_Ct"} : memref<4x4xi32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<4x4xi8>, %arg1: memref<4x4xi8>, %arg2: memref<4x4xi32>) attributes {dataflow, itypes = "sss", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [1], families = [#spmw.family<name = "feed_up_bind", type = memref<4xi8>, block = [4], depth = 16, shape = [1]>], ports = [#spmw.port_map<port = "chan", family = "feed_up_bind", kind = "table", slots = dense<0> : tensor<1xi32>>]> roles = [#spmw.role<unit = @feed_up_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<1xi32> : memref<4x4xi8>
    spmw.map(%arg1) topology = <grid = [1], families = [#spmw.family<name = "feed_3_up_bind", type = memref<4xi8>, block = [4], depth = 16, shape = [1]>], ports = [#spmw.port_map<port = "chan", family = "feed_3_up_bind", kind = "table", slots = dense<0> : tensor<1xi32>>]> roles = [#spmw.role<unit = @feed_3_up_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<1xi32> : memref<4x4xi8>
    spmw.map() topology = <grid = [4, 4], families = [#spmw.family<name = "pe_east_west", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "pe_south_north", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 2, shape = [4]>, #spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 2, shape = [4]>, #spmw.family<name = "drain_mine_bind", type = i32, block = [], depth = 2, shape = [16]>], ports = [#spmw.port_map<port = "lane", family = "pe_west_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "lane", family = "pe_north_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "c", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "mine", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @pe_r0, missing = ["c"], ports = ["c", "east", "north", "south", "west"]>, #spmw.role<unit = @pe_r1, missing = ["c", "west"], ports = ["c", "east", "north", "south", "west"]>, #spmw.role<unit = @pe_r2, missing = ["c", "south"], ports = ["c", "east", "north", "west"]>, #spmw.role<unit = @pe_r3, missing = ["c", "north"], ports = ["c", "east", "north", "south", "west"]>, #spmw.role<unit = @pe_r4, missing = ["c", "east"], ports = ["c", "north", "south", "west"]>, #spmw.role<unit = @pe_r5, missing = ["c", "south", "west"], ports = ["c", "east", "north", "west"]>, #spmw.role<unit = @pe_r6, missing = ["c", "north", "west"], ports = ["c", "east", "north", "south", "west"]>, #spmw.role<unit = @pe_r7, missing = ["c", "east", "south"], ports = ["c", "north", "west"]>, #spmw.role<unit = @pe_r8, missing = ["c", "east", "north"], ports = ["c", "north", "south", "west"]>] classes = dense<[6, 3, 3, 8, 1, 0, 0, 4, 1, 0, 0, 4, 5, 2, 2, 7]> : tensor<16xi32>
    spmw.map() topology = <grid = [4, 4], families = [#spmw.family<name = "drain_down_up", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "drain_mine_bind", type = i32, block = [], depth = 2, shape = [16]>, #spmw.family<name = "drain_down_bind", type = i32, block = [], depth = 4, shape = [4]>], ports = [#spmw.port_map<port = "c", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "mine", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "down", family = "drain_down_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>, #spmw.port_map<port = "down", family = "drain_down_up", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "up", family = "drain_down_up", kind = "affine", offset = [0, 0]>]> roles = [#spmw.role<unit = @drain_r0, missing = ["mine"], ports = ["down", "mine", "up"]>, #spmw.role<unit = @drain_r1, missing = ["mine", "up"], ports = ["down", "mine"]>, #spmw.role<unit = @drain_r2, missing = ["down", "mine"], ports = ["down", "mine", "up"]>] classes = dense<[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]> : tensor<16xi32>
    spmw.map() topology = <grid = [4], families = [#spmw.family<name = "feed_down_up", type = memref<4xi8>, block = [4], depth = 2, shape = [4]>, #spmw.family<name = "feed_up_bind", type = memref<4xi8>, block = [4], depth = 16, shape = [1]>, #spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "up", family = "feed_up_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "lane", family = "pe_west_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<-1> : tensor<4xi32>>, #spmw.port_map<port = "down", family = "feed_down_up", kind = "affine", offset = [1]>, #spmw.port_map<port = "up", family = "feed_down_up", kind = "affine", offset = [0]>]> roles = [#spmw.role<unit = @feed_r0, missing = ["lane"], ports = ["down", "lane", "up"]>, #spmw.role<unit = @feed_r1, missing = ["lane", "up"], ports = ["down", "lane", "up"]>, #spmw.role<unit = @feed_r2, missing = ["down", "lane"], ports = ["lane", "up"]>] classes = dense<[1, 0, 0, 2]> : tensor<4xi32>
    spmw.map() topology = <grid = [4], families = [#spmw.family<name = "feed_3_down_up", type = memref<4xi8>, block = [4], depth = 2, shape = [4]>, #spmw.family<name = "feed_3_up_bind", type = memref<4xi8>, block = [4], depth = 16, shape = [1]>, #spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "up", family = "feed_3_up_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "lane", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<-1> : tensor<4xi32>>, #spmw.port_map<port = "down", family = "feed_3_down_up", kind = "affine", offset = [1]>, #spmw.port_map<port = "up", family = "feed_3_down_up", kind = "affine", offset = [0]>]> roles = [#spmw.role<unit = @feed_3_r0, missing = ["lane"], ports = ["down", "lane", "up"]>, #spmw.role<unit = @feed_3_r1, missing = ["lane", "up"], ports = ["down", "lane", "up"]>, #spmw.role<unit = @feed_3_r2, missing = ["down", "lane"], ports = ["lane", "up"]>] classes = dense<[1, 0, 0, 2]> : tensor<4xi32>
    spmw.map(%arg2) topology = <grid = [4], families = [#spmw.family<name = "drain_down_bind", type = i32, block = [], depth = 4, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "drain_down_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @drain_down_drain, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<4x4xi32>
    return
  }
}
