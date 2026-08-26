#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @pe_west_load_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[0, %arg2] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_west_load_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[1, %arg2] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_west_load_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[2, %arg2] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_west_load_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[3, %arg2] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 0] {from = "local_B"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 1] {from = "local_B"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 2] {from = "local_B"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 3] {from = "local_B"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_0_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[0, 0] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_0_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[0, 1] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_0_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[0, 2] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_0_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[0, 3] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_1_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[1, 0] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_1_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[1, 1] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_1_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[1, 2] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_1_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[1, 3] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_2_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[2, 0] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_2_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[2, 1] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_2_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[2, 2] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_2_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
      allo.stream_put(%arg3, [], %13) : !allo.stream<i8, 2> contains i8
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<i32>
    affine.store %0, %arg0[2, 3] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_3_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
    affine.store %0, %arg0[3, 0] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_3_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
    affine.store %0, %arg0[3, 1] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_3_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
    affine.store %0, %arg0[3, 2] {to = "local_C"} : memref<4x4xi32, #map>
    return
  }
  func.func @pe_3_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s__", otypes = "", stypes = "_ii"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg3 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc_0 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %1, %alloc_0[] {to = "a"} : memref<i8>
      %2 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
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
    affine.store %0, %arg0[3, 3] {to = "local_C"} : memref<4x4xi32, #map>
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
    %32 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 2>
    %33 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 2>
    %34 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 2>
    %35 = allo.stream_construct() {name = "pe_west_bind_3"} : !allo.stream<i8, 2>
    %36 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 2>
    %37 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 2>
    %38 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 2>
    %39 = allo.stream_construct() {name = "pe_north_bind_3"} : !allo.stream<i8, 2>
    call @pe_west_load_0(%arg0, %32) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_west_load_1(%arg0, %33) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_west_load_2(%arg0, %34) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_west_load_3(%arg0, %35) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_0(%arg1, %36) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_1(%arg1, %37) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_2(%arg1, %38) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_3(%arg1, %39) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_0_0(%arg2, %32, %36, %1, %20) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_0_1(%arg2, %1, %37, %2, %21) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_0_2(%arg2, %2, %38, %3, %22) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_0_3(%arg2, %3, %39, %23) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_0(%arg2, %33, %20, %5, %24) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_1(%arg2, %5, %21, %6, %25) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_2(%arg2, %6, %22, %7, %26) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_3(%arg2, %7, %23, %27) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_0(%arg2, %34, %24, %9, %28) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_1(%arg2, %9, %25, %10, %29) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_2(%arg2, %10, %26, %11, %30) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_3(%arg2, %11, %27, %31) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_3_0(%arg2, %35, %28, %13) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_3_1(%arg2, %13, %29, %14) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_3_2(%arg2, %14, %30, %15) : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_3_3(%arg2, %15, %31) {last} : (memref<4x4xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    return
  }
}
