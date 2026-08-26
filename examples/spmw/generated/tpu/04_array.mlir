#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @mac_a_in_load_0(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 6 {
      %0 = affine.load %arg0[%arg2, 0] {from = "local_A"} : memref<6x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_1(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 6 {
      %0 = affine.load %arg0[%arg2, 1] {from = "local_A"} : memref<6x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_2(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 6 {
      %0 = affine.load %arg0[%arg2, 2] {from = "local_A"} : memref<6x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_3(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 6 {
      %0 = affine.load %arg0[%arg2, 3] {from = "local_A"} : memref<6x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_ioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[0, 0] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg2, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_ioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[0, 1] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg2, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_ioo"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[0, 2] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg2, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg3, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s__", otypes = "", stypes = "_io"} {
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[0, 3] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg2, [], %9) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[1, 0] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[1, 1] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[1, 2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[1, 3] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[2, 0] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[2, 1] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[2, 2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[2, 3] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[3, 0] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[3, 1] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iioo"} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[3, 2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[3, 3] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg3, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_0(%arg0: !allo.stream<i32, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg0, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc[] {to = "z"} : memref<i32>
      %1 = affine.load %alloc[] {from = "z"} : memref<i32>
      %2 = arith.cmpi slt, %1, %c0_i32 : i32
      scf.if %2 {
        affine.store %c0_i32, %alloc[] {to = "z"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "z"} : memref<i32>
      %4 = arith.shrsi %3, %c4_i32 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_0 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_0[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_0[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_1(%arg0: !allo.stream<i32, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg0, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc[] {to = "z"} : memref<i32>
      %1 = affine.load %alloc[] {from = "z"} : memref<i32>
      %2 = arith.cmpi slt, %1, %c0_i32 : i32
      scf.if %2 {
        affine.store %c0_i32, %alloc[] {to = "z"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "z"} : memref<i32>
      %4 = arith.shrsi %3, %c4_i32 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_0 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_0[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_0[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_2(%arg0: !allo.stream<i32, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg0, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc[] {to = "z"} : memref<i32>
      %1 = affine.load %alloc[] {from = "z"} : memref<i32>
      %2 = arith.cmpi slt, %1, %c0_i32 : i32
      scf.if %2 {
        affine.store %c0_i32, %alloc[] {to = "z"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "z"} : memref<i32>
      %4 = arith.shrsi %3, %c4_i32 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_0 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_0[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_0[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_3(%arg0: !allo.stream<i32, 2>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "io"} {
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg0, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc[] {to = "z"} : memref<i32>
      %1 = affine.load %alloc[] {from = "z"} : memref<i32>
      %2 = arith.cmpi slt, %1, %c0_i32 : i32
      scf.if %2 {
        affine.store %c0_i32, %alloc[] {to = "z"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "z"} : memref<i32>
      %4 = arith.shrsi %3, %c4_i32 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_0 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_0[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_0[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_y_out_drain_0(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      affine.store %0, %arg0[%arg2, 0] {to = "local_Y"} : memref<6x4xi8, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_y_out_drain_1(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      affine.store %0, %arg0[%arg2, 1] {to = "local_Y"} : memref<6x4xi8, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_y_out_drain_2(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      affine.store %0, %arg0[%arg2, 2] {to = "local_Y"} : memref<6x4xi8, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @act_y_out_drain_3(%arg0: memref<6x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i8, 2> -> i8
      affine.store %0, %arg0[%arg2, 3] {to = "local_Y"} : memref<6x4xi8, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<6x4xi8, #map>, %arg1: memref<4x4xi8, #map>, %arg2: memref<6x4xi8, #map>) attributes {dataflow, itypes = "sss", top} {
    %0 = allo.stream_construct() {name = "mac_a_out_a_in_0_0"} : !allo.stream<i8, 2>
    %1 = allo.stream_construct() {name = "mac_a_out_a_in_0_1"} : !allo.stream<i8, 2>
    %2 = allo.stream_construct() {name = "mac_a_out_a_in_0_2"} : !allo.stream<i8, 2>
    %3 = allo.stream_construct() {name = "mac_a_out_a_in_0_3"} : !allo.stream<i8, 2>
    %4 = allo.stream_construct() {name = "mac_a_out_a_in_1_0"} : !allo.stream<i8, 2>
    %5 = allo.stream_construct() {name = "mac_a_out_a_in_1_1"} : !allo.stream<i8, 2>
    %6 = allo.stream_construct() {name = "mac_a_out_a_in_1_2"} : !allo.stream<i8, 2>
    %7 = allo.stream_construct() {name = "mac_a_out_a_in_1_3"} : !allo.stream<i8, 2>
    %8 = allo.stream_construct() {name = "mac_a_out_a_in_2_0"} : !allo.stream<i8, 2>
    %9 = allo.stream_construct() {name = "mac_a_out_a_in_2_1"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "mac_a_out_a_in_2_2"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "mac_a_out_a_in_2_3"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "mac_a_out_a_in_3_0"} : !allo.stream<i8, 2>
    %13 = allo.stream_construct() {name = "mac_a_out_a_in_3_1"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "mac_a_out_a_in_3_2"} : !allo.stream<i8, 2>
    %15 = allo.stream_construct() {name = "mac_a_out_a_in_3_3"} : !allo.stream<i8, 2>
    %16 = allo.stream_construct() {name = "mac_p_out_p_in_0_0"} : !allo.stream<i32, 2>
    %17 = allo.stream_construct() {name = "mac_p_out_p_in_0_1"} : !allo.stream<i32, 2>
    %18 = allo.stream_construct() {name = "mac_p_out_p_in_0_2"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "mac_p_out_p_in_0_3"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "mac_p_out_p_in_1_0"} : !allo.stream<i32, 2>
    %21 = allo.stream_construct() {name = "mac_p_out_p_in_1_1"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "mac_p_out_p_in_1_2"} : !allo.stream<i32, 2>
    %23 = allo.stream_construct() {name = "mac_p_out_p_in_1_3"} : !allo.stream<i32, 2>
    %24 = allo.stream_construct() {name = "mac_p_out_p_in_2_0"} : !allo.stream<i32, 2>
    %25 = allo.stream_construct() {name = "mac_p_out_p_in_2_1"} : !allo.stream<i32, 2>
    %26 = allo.stream_construct() {name = "mac_p_out_p_in_2_2"} : !allo.stream<i32, 2>
    %27 = allo.stream_construct() {name = "mac_p_out_p_in_2_3"} : !allo.stream<i32, 2>
    %28 = allo.stream_construct() {name = "mac_p_out_p_in_3_0"} : !allo.stream<i32, 2>
    %29 = allo.stream_construct() {name = "mac_p_out_p_in_3_1"} : !allo.stream<i32, 2>
    %30 = allo.stream_construct() {name = "mac_p_out_p_in_3_2"} : !allo.stream<i32, 2>
    %31 = allo.stream_construct() {name = "mac_p_out_p_in_3_3"} : !allo.stream<i32, 2>
    %32 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 2>
    %33 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 2>
    %34 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 2>
    %35 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 2>
    %36 = allo.stream_construct() {name = "act_z_in_bind_0"} : !allo.stream<i32, 2>
    %37 = allo.stream_construct() {name = "act_z_in_bind_1"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "act_z_in_bind_2"} : !allo.stream<i32, 2>
    %39 = allo.stream_construct() {name = "act_z_in_bind_3"} : !allo.stream<i32, 2>
    %40 = allo.stream_construct() {name = "act_y_out_bind_0"} : !allo.stream<i8, 2>
    %41 = allo.stream_construct() {name = "act_y_out_bind_1"} : !allo.stream<i8, 2>
    %42 = allo.stream_construct() {name = "act_y_out_bind_2"} : !allo.stream<i8, 2>
    %43 = allo.stream_construct() {name = "act_y_out_bind_3"} : !allo.stream<i8, 2>
    call @mac_a_in_load_0(%arg0, %32) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_1(%arg0, %33) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_2(%arg0, %34) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_3(%arg0, %35) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_0_0(%arg1, %32, %20, %1) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_1(%arg1, %1, %21, %2) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_2(%arg1, %2, %22, %3) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_3(%arg1, %3, %23) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_0(%arg1, %33, %20, %24, %5) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_1(%arg1, %5, %21, %25, %6) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_2(%arg1, %6, %22, %26, %7) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_3(%arg1, %7, %23, %27) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_0(%arg1, %34, %24, %28, %9) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_1(%arg1, %9, %25, %29, %10) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_2(%arg1, %10, %26, %30, %11) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_3(%arg1, %11, %27, %31) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_0(%arg1, %35, %28, %36, %13) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_1(%arg1, %13, %29, %37, %14) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_2(%arg1, %14, %30, %38, %15) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_3(%arg1, %15, %31, %39) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @act_0(%36, %40) : (!allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @act_1(%37, %41) : (!allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @act_2(%38, %42) : (!allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @act_3(%39, %43) : (!allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @act_y_out_drain_0(%arg2, %40) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @act_y_out_drain_1(%arg2, %41) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @act_y_out_drain_2(%arg2, %42) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @act_y_out_drain_3(%arg2, %43) {last} : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    return
  }
}
