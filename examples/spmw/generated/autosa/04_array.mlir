#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @feed_up_load_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    %c0_i8 = arith.constant {name = "%c0_i8"} 0 : i8
    affine.for %arg2 = 0 to 4 {
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      affine.for %arg3 = 0 to 4 {
        affine.store %c0_i8, %alloc[%arg3] : memref<4xi8>
      }
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %arg0[%arg2, %arg3] {from = "local_At"} : memref<4x4xi8, #map>
        affine.store %0, %alloc[%arg3] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @feed_2_up_load_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    %c0_i8 = arith.constant {name = "%c0_i8"} 0 : i8
    affine.for %arg2 = 0 to 4 {
      %alloc = memref.alloc() {name = "_blk"} : memref<4xi8>
      affine.for %arg3 = 0 to 4 {
        affine.store %c0_i8, %alloc[%arg3] : memref<4xi8>
      }
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %arg0[%arg2, %arg3] {from = "local_Bt"} : memref<4x4xi8, #map>
        affine.store %0, %alloc[%arg3] {to = "_blk"} : memref<4xi8>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_0_0(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 0 {
      allo.stream_put(%arg4, [], %c0_i32) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_0_1(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 0 {
      allo.stream_put(%arg4, [], %c0_i32) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_0_2(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 0 {
      allo.stream_put(%arg4, [], %c0_i32) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_0_3(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg4 = 0 to 0 {
      allo.stream_put(%arg3, [], %c0_i32) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_1_0(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 1 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_1_1(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 1 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_1_2(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 1 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_1_3(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 1 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_2_0(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 2 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_2_1(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 2 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_2_2(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "______", otypes = "", stypes = "iioooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg6 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_1[] {from = "b"} : memref<i8>
      allo.stream_put(%arg3, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg4, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg6 = 0 to 2 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg4, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_2_3(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 2 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_3_0(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 3 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_3_1(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 3 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_3_2(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "iiooi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg2, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    allo.stream_put(%arg3, [], %0) : !allo.stream<i32, 2> contains i32
    affine.for %arg5 = 0 to 3 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_3_3(%arg0: !allo.stream<i8, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioi"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg0, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
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
    affine.for %arg4 = 0 to 3 {
      %1 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_i", op_name = "S__i_1", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_0(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_1(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[1] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_2(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[2] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_3(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[3] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_2_0(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[0] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_2_1(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[1] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_2_2(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<memref<4xi8>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "ioo"} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[2] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
      allo.stream_put(%arg2, [], %0) : !allo.stream<memref<4xi8>, 2> contains memref<4xi8>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @feed_2_3(%arg0: !allo.stream<memref<4xi8>, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg0, []) {name = "packed"} : !allo.stream<memref<4xi8>, 2> -> memref<4xi8>
      %1 = affine.load %0[3] {from = "packed"} : memref<4xi8>
      allo.stream_put(%arg1, [], %1) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_c_out_drain_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[0, %arg2] {to = "local_Ct"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_c_out_drain_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[1, %arg2] {to = "local_Ct"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_c_out_drain_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[2, %arg2] {to = "local_Ct"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_c_out_drain_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[3, %arg2] {to = "local_Ct"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<4x4xi8, #map>, %arg1: memref<4x4xi8, #map>, %arg2: memref<4x4xi32, #map>) attributes {dataflow, itypes = "sss", top} {
    %0 = allo.stream_construct() {name = "pe_east_west_0_0"} : !allo.stream<i8, 2>
    %1 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<i8, 2>
    %2 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<i8, 2>
    %3 = allo.stream_construct() {name = "pe_east_west_0_3"} : !allo.stream<i8, 2>
    %4 = allo.stream_construct() {name = "pe_east_west_1_0"} : !allo.stream<i8, 2>
    %5 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<i8, 2>
    %6 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<i8, 2>
    %7 = allo.stream_construct() {name = "pe_east_west_1_3"} : !allo.stream<i8, 2>
    %8 = allo.stream_construct() {name = "pe_east_west_2_0"} : !allo.stream<i8, 2>
    %9 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "pe_east_west_2_3"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "pe_east_west_3_0"} : !allo.stream<i8, 2>
    %13 = allo.stream_construct() {name = "pe_east_west_3_1"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "pe_east_west_3_2"} : !allo.stream<i8, 2>
    %15 = allo.stream_construct() {name = "pe_east_west_3_3"} : !allo.stream<i8, 2>
    %16 = allo.stream_construct() {name = "pe_south_north_0_0"} : !allo.stream<i8, 2>
    %17 = allo.stream_construct() {name = "pe_south_north_0_1"} : !allo.stream<i8, 2>
    %18 = allo.stream_construct() {name = "pe_south_north_0_2"} : !allo.stream<i8, 2>
    %19 = allo.stream_construct() {name = "pe_south_north_0_3"} : !allo.stream<i8, 2>
    %20 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<i8, 2>
    %22 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<i8, 2>
    %23 = allo.stream_construct() {name = "pe_south_north_1_3"} : !allo.stream<i8, 2>
    %24 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<i8, 2>
    %25 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<i8, 2>
    %26 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<i8, 2>
    %27 = allo.stream_construct() {name = "pe_south_north_2_3"} : !allo.stream<i8, 2>
    %28 = allo.stream_construct() {name = "pe_south_north_3_0"} : !allo.stream<i8, 2>
    %29 = allo.stream_construct() {name = "pe_south_north_3_1"} : !allo.stream<i8, 2>
    %30 = allo.stream_construct() {name = "pe_south_north_3_2"} : !allo.stream<i8, 2>
    %31 = allo.stream_construct() {name = "pe_south_north_3_3"} : !allo.stream<i8, 2>
    %32 = allo.stream_construct() {name = "pe_c_out_c_in_0_0"} : !allo.stream<i32, 2>
    %33 = allo.stream_construct() {name = "pe_c_out_c_in_0_1"} : !allo.stream<i32, 2>
    %34 = allo.stream_construct() {name = "pe_c_out_c_in_0_2"} : !allo.stream<i32, 2>
    %35 = allo.stream_construct() {name = "pe_c_out_c_in_0_3"} : !allo.stream<i32, 2>
    %36 = allo.stream_construct() {name = "pe_c_out_c_in_1_0"} : !allo.stream<i32, 2>
    %37 = allo.stream_construct() {name = "pe_c_out_c_in_1_1"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "pe_c_out_c_in_1_2"} : !allo.stream<i32, 2>
    %39 = allo.stream_construct() {name = "pe_c_out_c_in_1_3"} : !allo.stream<i32, 2>
    %40 = allo.stream_construct() {name = "pe_c_out_c_in_2_0"} : !allo.stream<i32, 2>
    %41 = allo.stream_construct() {name = "pe_c_out_c_in_2_1"} : !allo.stream<i32, 2>
    %42 = allo.stream_construct() {name = "pe_c_out_c_in_2_2"} : !allo.stream<i32, 2>
    %43 = allo.stream_construct() {name = "pe_c_out_c_in_2_3"} : !allo.stream<i32, 2>
    %44 = allo.stream_construct() {name = "pe_c_out_c_in_3_0"} : !allo.stream<i32, 2>
    %45 = allo.stream_construct() {name = "pe_c_out_c_in_3_1"} : !allo.stream<i32, 2>
    %46 = allo.stream_construct() {name = "pe_c_out_c_in_3_2"} : !allo.stream<i32, 2>
    %47 = allo.stream_construct() {name = "pe_c_out_c_in_3_3"} : !allo.stream<i32, 2>
    %48 = allo.stream_construct() {name = "feed_down_up_0"} : !allo.stream<memref<4xi8>, 2>
    %49 = allo.stream_construct() {name = "feed_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %50 = allo.stream_construct() {name = "feed_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %51 = allo.stream_construct() {name = "feed_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %52 = allo.stream_construct() {name = "feed_2_down_up_0"} : !allo.stream<memref<4xi8>, 2>
    %53 = allo.stream_construct() {name = "feed_2_down_up_1"} : !allo.stream<memref<4xi8>, 2>
    %54 = allo.stream_construct() {name = "feed_2_down_up_2"} : !allo.stream<memref<4xi8>, 2>
    %55 = allo.stream_construct() {name = "feed_2_down_up_3"} : !allo.stream<memref<4xi8>, 2>
    %56 = allo.stream_construct() {name = "feed_up_bind_0"} : !allo.stream<memref<4xi8>, 2>
    %57 = allo.stream_construct() {name = "feed_2_up_bind_0"} : !allo.stream<memref<4xi8>, 2>
    %58 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 2>
    %59 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 2>
    %60 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 2>
    %61 = allo.stream_construct() {name = "pe_west_bind_3"} : !allo.stream<i8, 2>
    %62 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 2>
    %63 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 2>
    %64 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 2>
    %65 = allo.stream_construct() {name = "pe_north_bind_3"} : !allo.stream<i8, 2>
    %66 = allo.stream_construct() {name = "pe_c_out_bind_0"} : !allo.stream<i32, 2>
    %67 = allo.stream_construct() {name = "pe_c_out_bind_1"} : !allo.stream<i32, 2>
    %68 = allo.stream_construct() {name = "pe_c_out_bind_2"} : !allo.stream<i32, 2>
    %69 = allo.stream_construct() {name = "pe_c_out_bind_3"} : !allo.stream<i32, 2>
    call @feed_up_load_0(%arg0, %56) : (memref<4x4xi8, #map>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_up_load_0(%arg1, %57) : (memref<4x4xi8, #map>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @pe_0_0(%58, %62, %1, %20, %36) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_0_1(%1, %63, %2, %21, %37) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_0_2(%2, %64, %3, %22, %38) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_0_3(%3, %65, %23, %39) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_1_0(%59, %20, %5, %24, %40, %36) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_1_1(%5, %21, %6, %25, %41, %37) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_1_2(%6, %22, %7, %26, %42, %38) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_1_3(%7, %23, %27, %43, %39) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_2_0(%60, %24, %9, %28, %44, %40) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_2_1(%9, %25, %10, %29, %45, %41) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_2_2(%10, %26, %11, %30, %46, %42) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_2_3(%11, %27, %31, %47, %43) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_3_0(%61, %28, %13, %66, %44) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_3_1(%13, %29, %14, %67, %45) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_3_2(%14, %30, %15, %68, %46) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @pe_3_3(%15, %31, %69, %47) : (!allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @feed_0(%56, %58, %49) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_1(%49, %59, %50) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2(%50, %60, %51) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_3(%51, %61) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>) -> ()
    call @feed_2_0(%57, %62, %53) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_1(%53, %63, %54) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_2(%54, %64, %55) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>, !allo.stream<memref<4xi8>, 2>) -> ()
    call @feed_2_3(%55, %65) : (!allo.stream<memref<4xi8>, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_c_out_drain_0(%arg2, %66) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @pe_c_out_drain_1(%arg2, %67) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @pe_c_out_drain_2(%arg2, %68) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @pe_c_out_drain_3(%arg2, %69) {last} : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    return
  }
}
