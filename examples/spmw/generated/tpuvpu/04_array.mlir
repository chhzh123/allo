#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map1 = affine_map<(d0) -> (d0, 0)>
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
  func.func @vpu_op_in_load_0(%arg0: memref<8xi32, #map1>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 8 {
      %0 = affine.load %arg0[%arg2] {from = "local_Prog"} : memref<8xi32, #map1>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
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
  func.func @vpu_0(%arg0: memref<4xi32, #map1>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg5 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg5] : memref<8xi32>
    }
    affine.for %arg5 = 0 to 8 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg6 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg6] : memref<4xi32>
      }
      affine.for %arg6 = 0 to 8 {
        %1 = affine.load %alloc[%arg6] {from = "prog"} : memref<8xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_2[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %3 = arith.shrsi %2, %c24_i32 : i32
        %4 = arith.andi %3, %c255_i32 : i32
        %alloc_3 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_3[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %6 = arith.shrsi %5, %c20_i32 : i32
        %7 = arith.andi %6, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_4[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %9 = arith.shrsi %8, %c16_i32 : i32
        %10 = arith.andi %9, %c15_i32 : i32
        %alloc_5 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_5[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %12 = arith.andi %11, %c65535_i32 : i32
        %alloc_6 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_6[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
        %14 = arith.cmpi eq, %13, %c1_i32 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
          %16 = arith.cmpi eq, %15, %c2_i32 : i32
          scf.if %16 {
            %17 = affine.load %arg0[0] {from = "local_Bias"} : memref<4xi32, #map1>
            %18 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
            %18 = arith.cmpi eq, %17, %c3_i32 : i32
            scf.if %18 {
              %19 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
              %20 = arith.cmpi eq, %19, %c4_i32 : i32
              scf.if %20 {
                %21 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                memref.store %30, %alloc_1[%22] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                %22 = arith.cmpi eq, %21, %c5_i32 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  memref.store %32, %alloc_1[%24] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                  %24 = arith.cmpi eq, %23, %c6_i32 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                    %26 = arith.cmpi eq, %25, %c7_i32 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      memref.store %31, %alloc_1[%28] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                      %28 = arith.cmpi eq, %27, %c8_i32 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                        %30 = arith.index_cast %29 : i32 to index
                        %31 = memref.load %alloc_1[%30] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg4, [], %31) : !allo.stream<i32, 2> contains i32
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "step", op_name = "S_step_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_1(%arg0: memref<4xi32, #map1>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg5 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg5] : memref<8xi32>
    }
    affine.for %arg5 = 0 to 8 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg6 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg6] : memref<4xi32>
      }
      affine.for %arg6 = 0 to 8 {
        %1 = affine.load %alloc[%arg6] {from = "prog"} : memref<8xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_2[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %3 = arith.shrsi %2, %c24_i32 : i32
        %4 = arith.andi %3, %c255_i32 : i32
        %alloc_3 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_3[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %6 = arith.shrsi %5, %c20_i32 : i32
        %7 = arith.andi %6, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_4[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %9 = arith.shrsi %8, %c16_i32 : i32
        %10 = arith.andi %9, %c15_i32 : i32
        %alloc_5 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_5[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %12 = arith.andi %11, %c65535_i32 : i32
        %alloc_6 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_6[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
        %14 = arith.cmpi eq, %13, %c1_i32 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
          %16 = arith.cmpi eq, %15, %c2_i32 : i32
          scf.if %16 {
            %17 = affine.load %arg0[1] {from = "local_Bias"} : memref<4xi32, #map1>
            %18 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
            %18 = arith.cmpi eq, %17, %c3_i32 : i32
            scf.if %18 {
              %19 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
              %20 = arith.cmpi eq, %19, %c4_i32 : i32
              scf.if %20 {
                %21 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                memref.store %30, %alloc_1[%22] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                %22 = arith.cmpi eq, %21, %c5_i32 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  memref.store %32, %alloc_1[%24] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                  %24 = arith.cmpi eq, %23, %c6_i32 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                    %26 = arith.cmpi eq, %25, %c7_i32 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      memref.store %31, %alloc_1[%28] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                      %28 = arith.cmpi eq, %27, %c8_i32 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                        %30 = arith.index_cast %29 : i32 to index
                        %31 = memref.load %alloc_1[%30] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg4, [], %31) : !allo.stream<i32, 2> contains i32
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "step", op_name = "S_step_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_2(%arg0: memref<4xi32, #map1>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg5 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg5] : memref<8xi32>
    }
    affine.for %arg5 = 0 to 8 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg6 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg6] : memref<4xi32>
      }
      affine.for %arg6 = 0 to 8 {
        %1 = affine.load %alloc[%arg6] {from = "prog"} : memref<8xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_2[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %3 = arith.shrsi %2, %c24_i32 : i32
        %4 = arith.andi %3, %c255_i32 : i32
        %alloc_3 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_3[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %6 = arith.shrsi %5, %c20_i32 : i32
        %7 = arith.andi %6, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_4[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %9 = arith.shrsi %8, %c16_i32 : i32
        %10 = arith.andi %9, %c15_i32 : i32
        %alloc_5 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_5[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %12 = arith.andi %11, %c65535_i32 : i32
        %alloc_6 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_6[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
        %14 = arith.cmpi eq, %13, %c1_i32 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
          %16 = arith.cmpi eq, %15, %c2_i32 : i32
          scf.if %16 {
            %17 = affine.load %arg0[2] {from = "local_Bias"} : memref<4xi32, #map1>
            %18 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
            %18 = arith.cmpi eq, %17, %c3_i32 : i32
            scf.if %18 {
              %19 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
              %20 = arith.cmpi eq, %19, %c4_i32 : i32
              scf.if %20 {
                %21 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                memref.store %30, %alloc_1[%22] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                %22 = arith.cmpi eq, %21, %c5_i32 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  memref.store %32, %alloc_1[%24] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                  %24 = arith.cmpi eq, %23, %c6_i32 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                    %26 = arith.cmpi eq, %25, %c7_i32 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      memref.store %31, %alloc_1[%28] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                      %28 = arith.cmpi eq, %27, %c8_i32 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                        %30 = arith.index_cast %29 : i32 to index
                        %31 = memref.load %alloc_1[%30] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg4, [], %31) : !allo.stream<i32, 2> contains i32
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "step", op_name = "S_step_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_3(%arg0: memref<4xi32, #map1>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg4 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg4] : memref<8xi32>
    }
    affine.for %arg4 = 0 to 8 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg4] {to = "prog"} : memref<8xi32>
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg4 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg5 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg5] : memref<4xi32>
      }
      affine.for %arg5 = 0 to 8 {
        %1 = affine.load %alloc[%arg5] {from = "prog"} : memref<8xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_2[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %3 = arith.shrsi %2, %c24_i32 : i32
        %4 = arith.andi %3, %c255_i32 : i32
        %alloc_3 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_3[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %6 = arith.shrsi %5, %c20_i32 : i32
        %7 = arith.andi %6, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_4[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %9 = arith.shrsi %8, %c16_i32 : i32
        %10 = arith.andi %9, %c15_i32 : i32
        %alloc_5 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_5[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %12 = arith.andi %11, %c65535_i32 : i32
        %alloc_6 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_6[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
        %14 = arith.cmpi eq, %13, %c1_i32 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
          %16 = arith.cmpi eq, %15, %c2_i32 : i32
          scf.if %16 {
            %17 = affine.load %arg0[3] {from = "local_Bias"} : memref<4xi32, #map1>
            %18 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
            %18 = arith.cmpi eq, %17, %c3_i32 : i32
            scf.if %18 {
              %19 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
              %20 = arith.cmpi eq, %19, %c4_i32 : i32
              scf.if %20 {
                %21 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                memref.store %30, %alloc_1[%22] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                %22 = arith.cmpi eq, %21, %c5_i32 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  memref.store %32, %alloc_1[%24] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                  %24 = arith.cmpi eq, %23, %c6_i32 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_5[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                    %26 = arith.cmpi eq, %25, %c7_i32 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_6[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      memref.store %31, %alloc_1[%28] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_3[] {from = "opcode"} : memref<i32>
                      %28 = arith.cmpi eq, %27, %c8_i32 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_4[] {from = "dst"} : memref<i32>
                        %30 = arith.index_cast %29 : i32 to index
                        %31 = memref.load %alloc_1[%30] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg3, [], %31) : !allo.stream<i32, 2> contains i32
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "step", op_name = "S_step_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_y_out_drain_0(%arg0: memref<6x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 0] {to = "local_Y"} : memref<6x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_1(%arg0: memref<6x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 1] {to = "local_Y"} : memref<6x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_2(%arg0: memref<6x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 2] {to = "local_Y"} : memref<6x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_3(%arg0: memref<6x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 6 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 3] {to = "local_Y"} : memref<6x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<6x4xi8, #map>, %arg1: memref<8xi32, #map1>, %arg2: memref<4x4xi8, #map>, %arg3: memref<4xi32, #map1>, %arg4: memref<6x4xi32, #map>) attributes {dataflow, itypes = "sssss", top} {
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
    %32 = allo.stream_construct() {name = "vpu_op_out_op_in_0"} : !allo.stream<i32, 2>
    %33 = allo.stream_construct() {name = "vpu_op_out_op_in_1"} : !allo.stream<i32, 2>
    %34 = allo.stream_construct() {name = "vpu_op_out_op_in_2"} : !allo.stream<i32, 2>
    %35 = allo.stream_construct() {name = "vpu_op_out_op_in_3"} : !allo.stream<i32, 2>
    %36 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 2>
    %37 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 2>
    %38 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 2>
    %39 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 2>
    %40 = allo.stream_construct() {name = "vpu_z_in_bind_0"} : !allo.stream<i32, 2>
    %41 = allo.stream_construct() {name = "vpu_z_in_bind_1"} : !allo.stream<i32, 2>
    %42 = allo.stream_construct() {name = "vpu_z_in_bind_2"} : !allo.stream<i32, 2>
    %43 = allo.stream_construct() {name = "vpu_z_in_bind_3"} : !allo.stream<i32, 2>
    %44 = allo.stream_construct() {name = "vpu_op_in_bind_0"} : !allo.stream<i32, 2>
    %45 = allo.stream_construct() {name = "vpu_y_out_bind_0"} : !allo.stream<i32, 2>
    %46 = allo.stream_construct() {name = "vpu_y_out_bind_1"} : !allo.stream<i32, 2>
    %47 = allo.stream_construct() {name = "vpu_y_out_bind_2"} : !allo.stream<i32, 2>
    %48 = allo.stream_construct() {name = "vpu_y_out_bind_3"} : !allo.stream<i32, 2>
    call @mac_a_in_load_0(%arg0, %36) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_1(%arg0, %37) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_2(%arg0, %38) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_3(%arg0, %39) : (memref<6x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @vpu_op_in_load_0(%arg1, %44) : (memref<8xi32, #map1>, !allo.stream<i32, 2>) -> ()
    call @mac_0_0(%arg2, %36, %20, %1) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_1(%arg2, %1, %21, %2) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_2(%arg2, %2, %22, %3) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_0_3(%arg2, %3, %23) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_0(%arg2, %37, %20, %24, %5) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_1(%arg2, %5, %21, %25, %6) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_2(%arg2, %6, %22, %26, %7) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_1_3(%arg2, %7, %23, %27) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_0(%arg2, %38, %24, %28, %9) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_1(%arg2, %9, %25, %29, %10) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_2(%arg2, %10, %26, %30, %11) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_2_3(%arg2, %11, %27, %31) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_0(%arg2, %39, %28, %40, %13) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_1(%arg2, %13, %29, %41, %14) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_2(%arg2, %14, %30, %42, %15) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>) -> ()
    call @mac_3_3(%arg2, %15, %31, %43) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_0(%arg3, %44, %33, %40, %45) : (memref<4xi32, #map1>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_1(%arg3, %33, %34, %41, %46) : (memref<4xi32, #map1>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_2(%arg3, %34, %35, %42, %47) : (memref<4xi32, #map1>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_3(%arg3, %35, %43, %48) : (memref<4xi32, #map1>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_0(%arg4, %45) : (memref<6x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_1(%arg4, %46) : (memref<6x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_2(%arg4, %47) : (memref<6x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_3(%arg4, %48) {last} : (memref<6x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    return
  }
}
