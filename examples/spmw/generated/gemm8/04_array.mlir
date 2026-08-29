#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @pe_west_load_0(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[0, %arg2] {from = "local_A"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_west_load_1(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[1, %arg2] {from = "local_A"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_west_load_2(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[2, %arg2] {from = "local_A"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_0(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[%arg2, 0] {from = "local_B"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_1(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[%arg2, 1] {from = "local_B"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load_2(%arg0: memref<3x3xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 3 {
      %0 = affine.load %arg0[%arg2, 2] {from = "local_B"} : memref<3x3xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_0_0(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 3 {
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
    affine.store %0, %arg0[0, 0] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_0_1(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 3 {
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
    affine.store %0, %arg0[0, 1] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_0_2(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 3 {
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
    affine.store %0, %arg0[0, 2] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_1_0(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 3 {
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
    affine.store %0, %arg0[1, 0] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_1_1(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg5 = 0 to 3 {
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
    affine.store %0, %arg0[1, 1] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_1_2(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 3 {
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
    affine.store %0, %arg0[1, 2] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_2_0(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 3 {
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
    affine.store %0, %arg0[2, 0] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_2_1(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg4 = 0 to 3 {
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
    affine.store %0, %arg0[2, 1] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @pe_2_2(%arg0: memref<3x3xi32, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s__", otypes = "", stypes = "_ii"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "acc"} : memref<i32>
    affine.store %c0_i32, %alloc[] {to = "acc"} : memref<i32>
    affine.for %arg3 = 0 to 3 {
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
    affine.store %0, %arg0[2, 2] {to = "local_C"} : memref<3x3xi32, #map>
    return
  }
  func.func @top(%arg0: memref<3x3xi8, #map>, %arg1: memref<3x3xi8, #map>, %arg2: memref<3x3xi32, #map>) attributes {dataflow, itypes = "sss", top} {
    %0 = allo.stream_construct() {name = "pe_east_west_0_0"} : !allo.stream<i8, 2>
    %1 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<i8, 2>
    %2 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<i8, 2>
    %3 = allo.stream_construct() {name = "pe_east_west_1_0"} : !allo.stream<i8, 2>
    %4 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<i8, 2>
    %5 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<i8, 2>
    %6 = allo.stream_construct() {name = "pe_east_west_2_0"} : !allo.stream<i8, 2>
    %7 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<i8, 2>
    %8 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<i8, 2>
    %9 = allo.stream_construct() {name = "pe_south_north_0_0"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "pe_south_north_0_1"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "pe_south_north_0_2"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<i8, 2>
    %13 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<i8, 2>
    %15 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<i8, 2>
    %16 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<i8, 2>
    %17 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<i8, 2>
    %18 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<i8, 2>
    %19 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<i8, 2>
    %20 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<i8, 2>
    %22 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<i8, 2>
    %23 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<i8, 2>
    call @pe_west_load_0(%arg0, %18) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_west_load_1(%arg0, %19) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_west_load_2(%arg0, %20) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_0(%arg1, %21) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_1(%arg1, %22) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_north_load_2(%arg1, %23) : (memref<3x3xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @pe_0_0(%arg2, %18, %21, %1, %12) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_0_1(%arg2, %1, %22, %2, %13) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_0_2(%arg2, %2, %23, %14) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_0(%arg2, %19, %12, %4, %15) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_1(%arg2, %4, %13, %5, %16) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_1_2(%arg2, %5, %14, %17) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_0(%arg2, %20, %15, %7) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_1(%arg2, %7, %16, %8) : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    call @pe_2_2(%arg2, %8, %17) {last} : (memref<3x3xi32, #map>, !allo.stream<i8, 2>, !allo.stream<i8, 2>) -> ()
    return
  }
}
