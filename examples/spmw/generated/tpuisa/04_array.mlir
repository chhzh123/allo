#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0, 0, 0)>
module {
  func.func @mac_a_in_load_0(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 0] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_1(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 1] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_2(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 2] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_a_in_load_3(%arg0: memref<4x4xi8, #map>, %arg1: !allo.stream<i8, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 3] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i8, 2> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_op_in_load_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 0] {from = "local_MProg"} : memref<4x4xi32, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_op_in_load_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 1] {from = "local_MProg"} : memref<4x4xi32, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_op_in_load_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 2] {from = "local_MProg"} : memref<4x4xi32, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_op_in_load_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.load %arg0[%arg2, 3] {from = "local_MProg"} : memref<4x4xi32, #map>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_op_in_load_0(%arg0: memref<16xi32, #map1>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_o"} {
    affine.for %arg2 = 0 to 16 {
      %0 = affine.load %arg0[%arg2] {from = "local_VProg"} : memref<16xi32, #map1>
      allo.stream_put(%arg1, [], %0) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_0(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = "", stypes = "_ioioo"} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg6 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %9) : !allo.stream<i8, 2> contains i8
      %10 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %arg0[%c0, %c0, %11] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %13 = arith.extsi %12 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "wt"} : memref<i32>
      %14 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %15 = arith.cmpi eq, %14, %c1_i32 : i32
      scf.if %15 {
        %16 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %19 = arith.extsi %17 : i8 to i40
        %20 = arith.extsi %18 : i32 to i40
        %21 = arith.muli %19, %20 : i40
        %22 = arith.extsi %16 : i32 to i41
        %23 = arith.extsi %21 : i40 to i41
        %24 = arith.addi %22, %23 : i41
        allo.stream_put(%arg5, [], %24) : !allo.stream<i32, 2> contains i41
      } else {
        %16 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %17 = arith.cmpi eq, %16, %c2_i32 : i32
        scf.if %17 {
          %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %20 = arith.extsi %18 : i8 to i40
          %21 = arith.extsi %19 : i32 to i40
          %22 = arith.muli %20, %21 : i40
          allo.stream_put(%arg5, [], %22) : !allo.stream<i32, 2> contains i40
        } else {
          %18 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg5, [], %18) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_1(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = "", stypes = "_ioioo"} {
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg6 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %9) : !allo.stream<i8, 2> contains i8
      %10 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %arg0[%c0, %c1, %11] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %13 = arith.extsi %12 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "wt"} : memref<i32>
      %14 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %15 = arith.cmpi eq, %14, %c1_i32 : i32
      scf.if %15 {
        %16 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %19 = arith.extsi %17 : i8 to i40
        %20 = arith.extsi %18 : i32 to i40
        %21 = arith.muli %19, %20 : i40
        %22 = arith.extsi %16 : i32 to i41
        %23 = arith.extsi %21 : i40 to i41
        %24 = arith.addi %22, %23 : i41
        allo.stream_put(%arg5, [], %24) : !allo.stream<i32, 2> contains i41
      } else {
        %16 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %17 = arith.cmpi eq, %16, %c2_i32 : i32
        scf.if %17 {
          %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %20 = arith.extsi %18 : i8 to i40
          %21 = arith.extsi %19 : i32 to i40
          %22 = arith.muli %20, %21 : i40
          allo.stream_put(%arg5, [], %22) : !allo.stream<i32, 2> contains i40
        } else {
          %18 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg5, [], %18) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_2(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = "", stypes = "_ioioo"} {
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg6 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %9) : !allo.stream<i8, 2> contains i8
      %10 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %arg0[%c0, %c2, %11] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %13 = arith.extsi %12 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "wt"} : memref<i32>
      %14 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %15 = arith.cmpi eq, %14, %c1_i32 : i32
      scf.if %15 {
        %16 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %19 = arith.extsi %17 : i8 to i40
        %20 = arith.extsi %18 : i32 to i40
        %21 = arith.muli %19, %20 : i40
        %22 = arith.extsi %16 : i32 to i41
        %23 = arith.extsi %21 : i40 to i41
        %24 = arith.addi %22, %23 : i41
        allo.stream_put(%arg5, [], %24) : !allo.stream<i32, 2> contains i41
      } else {
        %16 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %17 = arith.cmpi eq, %16, %c2_i32 : i32
        scf.if %17 {
          %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %20 = arith.extsi %18 : i8 to i40
          %21 = arith.extsi %19 : i32 to i40
          %22 = arith.muli %20, %21 : i40
          allo.stream_put(%arg5, [], %22) : !allo.stream<i32, 2> contains i40
        } else {
          %18 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg5, [], %18) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_0_3(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg4 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      %2 = arith.shrsi %1, %c24_i32 : i32
      %3 = arith.andi %2, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "opcode"} : memref<i32>
      %4 = affine.load %alloc[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c16_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "tile"} : memref<i32>
      %7 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %7, %alloc_2[] {to = "a"} : memref<i8>
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_3[] {to = "p"} : memref<i32>
      %8 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = memref.load %arg0[%c0, %c3, %9] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %11 = arith.extsi %10 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %11, %alloc_4[] {to = "wt"} : memref<i32>
      %12 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %13 = arith.cmpi eq, %12, %c1_i32 : i32
      scf.if %13 {
        %14 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %15 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %16 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %17 = arith.extsi %15 : i8 to i40
        %18 = arith.extsi %16 : i32 to i40
        %19 = arith.muli %17, %18 : i40
        %20 = arith.extsi %14 : i32 to i41
        %21 = arith.extsi %19 : i40 to i41
        %22 = arith.addi %20, %21 : i41
        allo.stream_put(%arg3, [], %22) : !allo.stream<i32, 2> contains i41
      } else {
        %14 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %15 = arith.cmpi eq, %14, %c2_i32 : i32
        scf.if %15 {
          %16 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %17 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %18 = arith.extsi %16 : i8 to i40
          %19 = arith.extsi %17 : i32 to i40
          %20 = arith.muli %18, %19 : i40
          allo.stream_put(%arg3, [], %20) : !allo.stream<i32, 2> contains i40
        } else {
          %16 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg3, [], %16) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_0(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c1, %c0, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_1(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c1, %c1, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_2(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c1, %c2, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_1_3(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iiio"} {
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg5 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      %2 = arith.shrsi %1, %c24_i32 : i32
      %3 = arith.andi %2, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "opcode"} : memref<i32>
      %4 = affine.load %alloc[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c16_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "tile"} : memref<i32>
      %7 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %7, %alloc_2[] {to = "a"} : memref<i8>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %8, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg0[%c1, %c3, %10] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %12 = arith.extsi %11 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %12, %alloc_4[] {to = "wt"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %14 = arith.cmpi eq, %13, %c1_i32 : i32
      scf.if %14 {
        %15 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %16 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %17 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %18 = arith.extsi %16 : i8 to i40
        %19 = arith.extsi %17 : i32 to i40
        %20 = arith.muli %18, %19 : i40
        %21 = arith.extsi %15 : i32 to i41
        %22 = arith.extsi %20 : i40 to i41
        %23 = arith.addi %21, %22 : i41
        allo.stream_put(%arg4, [], %23) : !allo.stream<i32, 2> contains i41
      } else {
        %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %16 = arith.cmpi eq, %15, %c2_i32 : i32
        scf.if %16 {
          %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %19 = arith.extsi %17 : i8 to i40
          %20 = arith.extsi %18 : i32 to i40
          %21 = arith.muli %19, %20 : i40
          allo.stream_put(%arg4, [], %21) : !allo.stream<i32, 2> contains i40
        } else {
          %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg4, [], %17) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_0(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c2, %c0, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_1(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c2, %c1, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_2(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c2, %c2, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_2_3(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iiio"} {
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg5 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      %2 = arith.shrsi %1, %c24_i32 : i32
      %3 = arith.andi %2, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "opcode"} : memref<i32>
      %4 = affine.load %alloc[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c16_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "tile"} : memref<i32>
      %7 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %7, %alloc_2[] {to = "a"} : memref<i8>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %8, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg0[%c2, %c3, %10] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %12 = arith.extsi %11 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %12, %alloc_4[] {to = "wt"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %14 = arith.cmpi eq, %13, %c1_i32 : i32
      scf.if %14 {
        %15 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %16 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %17 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %18 = arith.extsi %16 : i8 to i40
        %19 = arith.extsi %17 : i32 to i40
        %20 = arith.muli %18, %19 : i40
        %21 = arith.extsi %15 : i32 to i41
        %22 = arith.extsi %20 : i40 to i41
        %23 = arith.addi %21, %22 : i41
        allo.stream_put(%arg4, [], %23) : !allo.stream<i32, 2> contains i41
      } else {
        %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %16 = arith.cmpi eq, %15, %c2_i32 : i32
        scf.if %16 {
          %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %19 = arith.extsi %17 : i8 to i40
          %20 = arith.extsi %18 : i32 to i40
          %21 = arith.muli %19, %20 : i40
          allo.stream_put(%arg4, [], %21) : !allo.stream<i32, 2> contains i40
        } else {
          %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg4, [], %17) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_0(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c3, %c0, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_1(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c3, %c1, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_2(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i8, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = "", stypes = "_ioiioo"} {
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg7 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %1) : !allo.stream<i32, 2> contains i32
      %2 = affine.load %alloc[] {from = "word"} : memref<i32>
      %3 = arith.shrsi %2, %c24_i32 : i32
      %4 = arith.andi %3, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "opcode"} : memref<i32>
      %5 = affine.load %alloc[] {from = "word"} : memref<i32>
      %6 = arith.shrsi %5, %c16_i32 : i32
      %7 = arith.andi %6, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %7, %alloc_1[] {to = "tile"} : memref<i32>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %8, %alloc_2[] {to = "a"} : memref<i8>
      %9 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %9, %alloc_3[] {to = "p"} : memref<i32>
      %10 = affine.load %alloc_2[] {from = "a"} : memref<i8>
      allo.stream_put(%arg5, [], %10) : !allo.stream<i8, 2> contains i8
      %11 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%c3, %c2, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_4[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg6, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg6, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_3_3(%arg0: memref<4x4x4xi8, #map2>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i8, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_iiio"} {
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    affine.for %arg5 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc[] {from = "word"} : memref<i32>
      %2 = arith.shrsi %1, %c24_i32 : i32
      %3 = arith.andi %2, %c255_i32 : i32
      %alloc_0 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "opcode"} : memref<i32>
      %4 = affine.load %alloc[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c16_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "tile"} : memref<i32>
      %7 = allo.stream_get(%arg2, []) : !allo.stream<i8, 2> -> i8
      %alloc_2 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %7, %alloc_2[] {to = "a"} : memref<i8>
      %8 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
      %alloc_3 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %8, %alloc_3[] {to = "p"} : memref<i32>
      %9 = affine.load %alloc_1[] {from = "tile"} : memref<i32>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg0[%c3, %c3, %10] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %12 = arith.extsi %11 : i8 to i32
      %alloc_4 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %12, %alloc_4[] {to = "wt"} : memref<i32>
      %13 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
      %14 = arith.cmpi eq, %13, %c1_i32 : i32
      scf.if %14 {
        %15 = affine.load %alloc_3[] {from = "p"} : memref<i32>
        %16 = affine.load %alloc_2[] {from = "a"} : memref<i8>
        %17 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
        %18 = arith.extsi %16 : i8 to i40
        %19 = arith.extsi %17 : i32 to i40
        %20 = arith.muli %18, %19 : i40
        %21 = arith.extsi %15 : i32 to i41
        %22 = arith.extsi %20 : i40 to i41
        %23 = arith.addi %21, %22 : i41
        allo.stream_put(%arg4, [], %23) : !allo.stream<i32, 2> contains i41
      } else {
        %15 = affine.load %alloc_0[] {from = "opcode"} : memref<i32>
        %16 = arith.cmpi eq, %15, %c2_i32 : i32
        scf.if %16 {
          %17 = affine.load %alloc_2[] {from = "a"} : memref<i8>
          %18 = affine.load %alloc_4[] {from = "wt"} : memref<i32>
          %19 = arith.extsi %17 : i8 to i40
          %20 = arith.extsi %18 : i32 to i40
          %21 = arith.muli %19, %20 : i40
          allo.stream_put(%arg4, [], %21) : !allo.stream<i32, 2> contains i40
        } else {
          %17 = affine.load %alloc_3[] {from = "p"} : memref<i32>
          allo.stream_put(%arg4, [], %17) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_0(%arg0: memref<4x2xi32, #map>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c12_i32 = arith.constant {name = "%c12_i32"} 12 : i32
    %c30_i32 = arith.constant {name = "%c30_i32"} 30 : i32
    %c11_i32 = arith.constant {name = "%c11_i32"} 11 : i32
    %c10_i32 = arith.constant {name = "%c10_i32"} 10 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c9_i32 = arith.constant {name = "%c9_i32"} 9 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg5 = 0 to 16 {
      affine.store %c0_i32, %alloc[%arg5] : memref<16xi32>
    }
    affine.for %arg5 = 0 to 16 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_1 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_1[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<16xi32>
      %2 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
    affine.for %arg5 = 0 to 4 {
      affine.store %c0_i32, %alloc_0[%arg5] : memref<4xi32>
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 16 {
        %0 = affine.load %alloc[%arg6] {from = "prog"} : memref<16xi32>
        %alloc_1 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_1[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %2 = arith.shrsi %1, %c24_i32 : i32
        %3 = arith.andi %2, %c255_i32 : i32
        %alloc_2 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_2[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %5 = arith.shrsi %4, %c20_i32 : i32
        %6 = arith.andi %5, %c15_i32 : i32
        %alloc_3 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_3[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %8 = arith.shrsi %7, %c16_i32 : i32
        %9 = arith.andi %8, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_4[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %11 = arith.andi %10, %c65535_i32 : i32
        %alloc_5 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_5[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
        %13 = arith.cmpi eq, %12, %c9_i32 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
          %alloc_6 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_6[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_6[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          memref.store %22, %alloc_0[%16] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
          %15 = arith.cmpi eq, %14, %c1_i32 : i32
          scf.if %15 {
            %16 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
            %alloc_6 = memref.alloc() {name = "z2"} : memref<i32>
            affine.store %16, %alloc_6[] {to = "z2"} : memref<i32>
            %17 = affine.load %alloc_6[] {from = "z2"} : memref<i32>
            %18 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_0[%19] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
            %17 = arith.cmpi eq, %16, %c2_i32 : i32
            scf.if %17 {
              %18 = affine.load %alloc_4[] {from = "src"} : memref<i32>
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg0[%c0, %19] {from = "local_Bias"} : memref<4x2xi32, #map>
              %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
              %22 = arith.index_cast %21 : i32 to index
              memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
              %19 = arith.cmpi eq, %18, %c3_i32 : i32
              scf.if %19 {
                %20 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                %21 = arith.cmpi eq, %20, %c4_i32 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i33
                  %29 = arith.extsi %27 : i32 to i33
                  %30 = arith.addi %28, %29 : i33
                  %31 = arith.trunci %30 : i33 to i32
                  memref.store %31, %alloc_0[%23] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                  %23 = arith.cmpi eq, %22, %c5_i32 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.extsi %26 : i32 to i64
                    %31 = arith.extsi %29 : i32 to i64
                    %32 = arith.muli %30, %31 : i64
                    %33 = arith.trunci %32 : i64 to i32
                    memref.store %33, %alloc_0[%25] {to = "reg"} : memref<4xi32>
                  } else {
                    %24 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                    %25 = arith.cmpi eq, %24, %c6_i32 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                      %30 = arith.index_cast %29 : i32 to index
                      %31 = memref.load %alloc_0[%30] {from = "reg"} : memref<4xi32>
                      %32 = arith.cmpi sgt, %28, %31 : i32
                      scf.if %32 {
                        %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                        %34 = arith.index_cast %33 : i32 to index
                        %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                        %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %37 = arith.index_cast %36 : i32 to index
                        memref.store %35, %alloc_0[%37] {to = "reg"} : memref<4xi32>
                      }
                    } else {
                      %26 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                      %27 = arith.cmpi eq, %26, %c7_i32 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        %31 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                        %32 = arith.shrsi %30, %31 : i32
                        memref.store %32, %alloc_0[%29] {to = "reg"} : memref<4xi32>
                      } else {
                        %28 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                        %29 = arith.cmpi eq, %28, %c10_i32 : i32
                        scf.if %29 {
                          %30 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                          %31 = arith.index_cast %30 : i32 to index
                          %32 = memref.load %alloc_0[%31] {from = "reg"} : memref<4xi32>
                          %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                          %34 = arith.index_cast %33 : i32 to index
                          %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                          %36 = arith.extsi %32 : i32 to i33
                          %37 = arith.extsi %35 : i32 to i33
                          %38 = arith.subi %36, %37 : i33
                          %39 = arith.trunci %38 : i33 to i32
                          memref.store %39, %alloc_0[%31] {to = "reg"} : memref<4xi32>
                        } else {
                          %30 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                          %31 = arith.cmpi eq, %30, %c11_i32 : i32
                          scf.if %31 {
                            %32 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %33 = arith.index_cast %32 : i32 to index
                            %34 = memref.load %alloc_0[%33] {from = "reg"} : memref<4xi32>
                            %alloc_6 = memref.alloc() {name = "e"} : memref<i32>
                            affine.store %34, %alloc_6[] {to = "e"} : memref<i32>
                            %35 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %36 = arith.cmpi slt, %35, %c0_i32 : i32
                            scf.if %36 {
                              affine.store %c0_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %37 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %38 = arith.cmpi sgt, %37, %c30_i32 : i32
                            scf.if %38 {
                              affine.store %c30_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %39 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %40 = arith.shli %c1_i32, %39 : i32
                            %41 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %42 = arith.index_cast %41 : i32 to index
                            memref.store %40, %alloc_0[%42] {to = "reg"} : memref<4xi32>
                          } else {
                            %32 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                            %33 = arith.cmpi eq, %32, %c12_i32 : i32
                            scf.if %33 {
                              %34 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                              %35 = arith.index_cast %34 : i32 to index
                              %36 = memref.load %alloc_0[%35] {from = "reg"} : memref<4xi32>
                              %alloc_6 = memref.alloc() {name = "d"} : memref<i32>
                              affine.store %36, %alloc_6[] {to = "d"} : memref<i32>
                              %37 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                              %38 = arith.cmpi sgt, %37, %c0_i32 : i32
                              scf.if %38 {
                                %39 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                                %40 = arith.shli %c1_i32, %39 : i32
                                %41 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                                %42 = arith.floordivsi %40, %41 : i32
                                %43 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %44 = arith.index_cast %43 : i32 to index
                                memref.store %42, %alloc_0[%44] {to = "reg"} : memref<4xi32>
                              } else {
                                %39 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %40 = arith.index_cast %39 : i32 to index
                                memref.store %c0_i32, %alloc_0[%40] {to = "reg"} : memref<4xi32>
                              }
                            } else {
                              %34 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                              %35 = arith.cmpi eq, %34, %c8_i32 : i32
                              scf.if %35 {
                                %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %37 = arith.index_cast %36 : i32 to index
                                %38 = memref.load %alloc_0[%37] {from = "reg"} : memref<4xi32>
                                allo.stream_put(%arg4, [], %38) : !allo.stream<i32, 2> contains i32
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "pc2", op_name = "S_pc2_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_1(%arg0: memref<4x2xi32, #map>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c12_i32 = arith.constant {name = "%c12_i32"} 12 : i32
    %c30_i32 = arith.constant {name = "%c30_i32"} 30 : i32
    %c11_i32 = arith.constant {name = "%c11_i32"} 11 : i32
    %c10_i32 = arith.constant {name = "%c10_i32"} 10 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c9_i32 = arith.constant {name = "%c9_i32"} 9 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg5 = 0 to 16 {
      affine.store %c0_i32, %alloc[%arg5] : memref<16xi32>
    }
    affine.for %arg5 = 0 to 16 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_1 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_1[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<16xi32>
      %2 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
    affine.for %arg5 = 0 to 4 {
      affine.store %c0_i32, %alloc_0[%arg5] : memref<4xi32>
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 16 {
        %0 = affine.load %alloc[%arg6] {from = "prog"} : memref<16xi32>
        %alloc_1 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_1[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %2 = arith.shrsi %1, %c24_i32 : i32
        %3 = arith.andi %2, %c255_i32 : i32
        %alloc_2 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_2[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %5 = arith.shrsi %4, %c20_i32 : i32
        %6 = arith.andi %5, %c15_i32 : i32
        %alloc_3 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_3[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %8 = arith.shrsi %7, %c16_i32 : i32
        %9 = arith.andi %8, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_4[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %11 = arith.andi %10, %c65535_i32 : i32
        %alloc_5 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_5[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
        %13 = arith.cmpi eq, %12, %c9_i32 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
          %alloc_6 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_6[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_6[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          memref.store %22, %alloc_0[%16] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
          %15 = arith.cmpi eq, %14, %c1_i32 : i32
          scf.if %15 {
            %16 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
            %alloc_6 = memref.alloc() {name = "z2"} : memref<i32>
            affine.store %16, %alloc_6[] {to = "z2"} : memref<i32>
            %17 = affine.load %alloc_6[] {from = "z2"} : memref<i32>
            %18 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_0[%19] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
            %17 = arith.cmpi eq, %16, %c2_i32 : i32
            scf.if %17 {
              %18 = affine.load %alloc_4[] {from = "src"} : memref<i32>
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg0[%c1, %19] {from = "local_Bias"} : memref<4x2xi32, #map>
              %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
              %22 = arith.index_cast %21 : i32 to index
              memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
              %19 = arith.cmpi eq, %18, %c3_i32 : i32
              scf.if %19 {
                %20 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                %21 = arith.cmpi eq, %20, %c4_i32 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i33
                  %29 = arith.extsi %27 : i32 to i33
                  %30 = arith.addi %28, %29 : i33
                  %31 = arith.trunci %30 : i33 to i32
                  memref.store %31, %alloc_0[%23] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                  %23 = arith.cmpi eq, %22, %c5_i32 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.extsi %26 : i32 to i64
                    %31 = arith.extsi %29 : i32 to i64
                    %32 = arith.muli %30, %31 : i64
                    %33 = arith.trunci %32 : i64 to i32
                    memref.store %33, %alloc_0[%25] {to = "reg"} : memref<4xi32>
                  } else {
                    %24 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                    %25 = arith.cmpi eq, %24, %c6_i32 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                      %30 = arith.index_cast %29 : i32 to index
                      %31 = memref.load %alloc_0[%30] {from = "reg"} : memref<4xi32>
                      %32 = arith.cmpi sgt, %28, %31 : i32
                      scf.if %32 {
                        %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                        %34 = arith.index_cast %33 : i32 to index
                        %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                        %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %37 = arith.index_cast %36 : i32 to index
                        memref.store %35, %alloc_0[%37] {to = "reg"} : memref<4xi32>
                      }
                    } else {
                      %26 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                      %27 = arith.cmpi eq, %26, %c7_i32 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        %31 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                        %32 = arith.shrsi %30, %31 : i32
                        memref.store %32, %alloc_0[%29] {to = "reg"} : memref<4xi32>
                      } else {
                        %28 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                        %29 = arith.cmpi eq, %28, %c10_i32 : i32
                        scf.if %29 {
                          %30 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                          %31 = arith.index_cast %30 : i32 to index
                          %32 = memref.load %alloc_0[%31] {from = "reg"} : memref<4xi32>
                          %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                          %34 = arith.index_cast %33 : i32 to index
                          %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                          %36 = arith.extsi %32 : i32 to i33
                          %37 = arith.extsi %35 : i32 to i33
                          %38 = arith.subi %36, %37 : i33
                          %39 = arith.trunci %38 : i33 to i32
                          memref.store %39, %alloc_0[%31] {to = "reg"} : memref<4xi32>
                        } else {
                          %30 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                          %31 = arith.cmpi eq, %30, %c11_i32 : i32
                          scf.if %31 {
                            %32 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %33 = arith.index_cast %32 : i32 to index
                            %34 = memref.load %alloc_0[%33] {from = "reg"} : memref<4xi32>
                            %alloc_6 = memref.alloc() {name = "e"} : memref<i32>
                            affine.store %34, %alloc_6[] {to = "e"} : memref<i32>
                            %35 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %36 = arith.cmpi slt, %35, %c0_i32 : i32
                            scf.if %36 {
                              affine.store %c0_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %37 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %38 = arith.cmpi sgt, %37, %c30_i32 : i32
                            scf.if %38 {
                              affine.store %c30_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %39 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %40 = arith.shli %c1_i32, %39 : i32
                            %41 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %42 = arith.index_cast %41 : i32 to index
                            memref.store %40, %alloc_0[%42] {to = "reg"} : memref<4xi32>
                          } else {
                            %32 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                            %33 = arith.cmpi eq, %32, %c12_i32 : i32
                            scf.if %33 {
                              %34 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                              %35 = arith.index_cast %34 : i32 to index
                              %36 = memref.load %alloc_0[%35] {from = "reg"} : memref<4xi32>
                              %alloc_6 = memref.alloc() {name = "d"} : memref<i32>
                              affine.store %36, %alloc_6[] {to = "d"} : memref<i32>
                              %37 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                              %38 = arith.cmpi sgt, %37, %c0_i32 : i32
                              scf.if %38 {
                                %39 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                                %40 = arith.shli %c1_i32, %39 : i32
                                %41 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                                %42 = arith.floordivsi %40, %41 : i32
                                %43 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %44 = arith.index_cast %43 : i32 to index
                                memref.store %42, %alloc_0[%44] {to = "reg"} : memref<4xi32>
                              } else {
                                %39 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %40 = arith.index_cast %39 : i32 to index
                                memref.store %c0_i32, %alloc_0[%40] {to = "reg"} : memref<4xi32>
                              }
                            } else {
                              %34 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                              %35 = arith.cmpi eq, %34, %c8_i32 : i32
                              scf.if %35 {
                                %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %37 = arith.index_cast %36 : i32 to index
                                %38 = memref.load %alloc_0[%37] {from = "reg"} : memref<4xi32>
                                allo.stream_put(%arg4, [], %38) : !allo.stream<i32, 2> contains i32
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "pc2", op_name = "S_pc2_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_2(%arg0: memref<4x2xi32, #map>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = "", stypes = "_ioio"} {
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c12_i32 = arith.constant {name = "%c12_i32"} 12 : i32
    %c30_i32 = arith.constant {name = "%c30_i32"} 30 : i32
    %c11_i32 = arith.constant {name = "%c11_i32"} 11 : i32
    %c10_i32 = arith.constant {name = "%c10_i32"} 10 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c9_i32 = arith.constant {name = "%c9_i32"} 9 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg5 = 0 to 16 {
      affine.store %c0_i32, %alloc[%arg5] : memref<16xi32>
    }
    affine.for %arg5 = 0 to 16 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_1 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_1[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<16xi32>
      %2 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      allo.stream_put(%arg2, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
    affine.for %arg5 = 0 to 4 {
      affine.store %c0_i32, %alloc_0[%arg5] : memref<4xi32>
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 16 {
        %0 = affine.load %alloc[%arg6] {from = "prog"} : memref<16xi32>
        %alloc_1 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_1[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %2 = arith.shrsi %1, %c24_i32 : i32
        %3 = arith.andi %2, %c255_i32 : i32
        %alloc_2 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_2[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %5 = arith.shrsi %4, %c20_i32 : i32
        %6 = arith.andi %5, %c15_i32 : i32
        %alloc_3 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_3[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %8 = arith.shrsi %7, %c16_i32 : i32
        %9 = arith.andi %8, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_4[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %11 = arith.andi %10, %c65535_i32 : i32
        %alloc_5 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_5[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
        %13 = arith.cmpi eq, %12, %c9_i32 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
          %alloc_6 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_6[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_6[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          memref.store %22, %alloc_0[%16] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
          %15 = arith.cmpi eq, %14, %c1_i32 : i32
          scf.if %15 {
            %16 = allo.stream_get(%arg3, []) : !allo.stream<i32, 2> -> i32
            %alloc_6 = memref.alloc() {name = "z2"} : memref<i32>
            affine.store %16, %alloc_6[] {to = "z2"} : memref<i32>
            %17 = affine.load %alloc_6[] {from = "z2"} : memref<i32>
            %18 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_0[%19] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
            %17 = arith.cmpi eq, %16, %c2_i32 : i32
            scf.if %17 {
              %18 = affine.load %alloc_4[] {from = "src"} : memref<i32>
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg0[%c2, %19] {from = "local_Bias"} : memref<4x2xi32, #map>
              %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
              %22 = arith.index_cast %21 : i32 to index
              memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
              %19 = arith.cmpi eq, %18, %c3_i32 : i32
              scf.if %19 {
                %20 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                %21 = arith.cmpi eq, %20, %c4_i32 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i33
                  %29 = arith.extsi %27 : i32 to i33
                  %30 = arith.addi %28, %29 : i33
                  %31 = arith.trunci %30 : i33 to i32
                  memref.store %31, %alloc_0[%23] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                  %23 = arith.cmpi eq, %22, %c5_i32 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.extsi %26 : i32 to i64
                    %31 = arith.extsi %29 : i32 to i64
                    %32 = arith.muli %30, %31 : i64
                    %33 = arith.trunci %32 : i64 to i32
                    memref.store %33, %alloc_0[%25] {to = "reg"} : memref<4xi32>
                  } else {
                    %24 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                    %25 = arith.cmpi eq, %24, %c6_i32 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                      %30 = arith.index_cast %29 : i32 to index
                      %31 = memref.load %alloc_0[%30] {from = "reg"} : memref<4xi32>
                      %32 = arith.cmpi sgt, %28, %31 : i32
                      scf.if %32 {
                        %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                        %34 = arith.index_cast %33 : i32 to index
                        %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                        %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %37 = arith.index_cast %36 : i32 to index
                        memref.store %35, %alloc_0[%37] {to = "reg"} : memref<4xi32>
                      }
                    } else {
                      %26 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                      %27 = arith.cmpi eq, %26, %c7_i32 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        %31 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                        %32 = arith.shrsi %30, %31 : i32
                        memref.store %32, %alloc_0[%29] {to = "reg"} : memref<4xi32>
                      } else {
                        %28 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                        %29 = arith.cmpi eq, %28, %c10_i32 : i32
                        scf.if %29 {
                          %30 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                          %31 = arith.index_cast %30 : i32 to index
                          %32 = memref.load %alloc_0[%31] {from = "reg"} : memref<4xi32>
                          %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                          %34 = arith.index_cast %33 : i32 to index
                          %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                          %36 = arith.extsi %32 : i32 to i33
                          %37 = arith.extsi %35 : i32 to i33
                          %38 = arith.subi %36, %37 : i33
                          %39 = arith.trunci %38 : i33 to i32
                          memref.store %39, %alloc_0[%31] {to = "reg"} : memref<4xi32>
                        } else {
                          %30 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                          %31 = arith.cmpi eq, %30, %c11_i32 : i32
                          scf.if %31 {
                            %32 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %33 = arith.index_cast %32 : i32 to index
                            %34 = memref.load %alloc_0[%33] {from = "reg"} : memref<4xi32>
                            %alloc_6 = memref.alloc() {name = "e"} : memref<i32>
                            affine.store %34, %alloc_6[] {to = "e"} : memref<i32>
                            %35 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %36 = arith.cmpi slt, %35, %c0_i32 : i32
                            scf.if %36 {
                              affine.store %c0_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %37 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %38 = arith.cmpi sgt, %37, %c30_i32 : i32
                            scf.if %38 {
                              affine.store %c30_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %39 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %40 = arith.shli %c1_i32, %39 : i32
                            %41 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %42 = arith.index_cast %41 : i32 to index
                            memref.store %40, %alloc_0[%42] {to = "reg"} : memref<4xi32>
                          } else {
                            %32 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                            %33 = arith.cmpi eq, %32, %c12_i32 : i32
                            scf.if %33 {
                              %34 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                              %35 = arith.index_cast %34 : i32 to index
                              %36 = memref.load %alloc_0[%35] {from = "reg"} : memref<4xi32>
                              %alloc_6 = memref.alloc() {name = "d"} : memref<i32>
                              affine.store %36, %alloc_6[] {to = "d"} : memref<i32>
                              %37 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                              %38 = arith.cmpi sgt, %37, %c0_i32 : i32
                              scf.if %38 {
                                %39 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                                %40 = arith.shli %c1_i32, %39 : i32
                                %41 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                                %42 = arith.floordivsi %40, %41 : i32
                                %43 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %44 = arith.index_cast %43 : i32 to index
                                memref.store %42, %alloc_0[%44] {to = "reg"} : memref<4xi32>
                              } else {
                                %39 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %40 = arith.index_cast %39 : i32 to index
                                memref.store %c0_i32, %alloc_0[%40] {to = "reg"} : memref<4xi32>
                              }
                            } else {
                              %34 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                              %35 = arith.cmpi eq, %34, %c8_i32 : i32
                              scf.if %35 {
                                %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %37 = arith.index_cast %36 : i32 to index
                                %38 = memref.load %alloc_0[%37] {from = "reg"} : memref<4xi32>
                                allo.stream_put(%arg4, [], %38) : !allo.stream<i32, 2> contains i32
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "pc2", op_name = "S_pc2_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_3(%arg0: memref<4x2xi32, #map>, %arg1: !allo.stream<i32, 2>, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s___", otypes = "", stypes = "_iio"} {
    %c3 = arith.constant {name = "%c3"} 3 : index
    %c8_i32 = arith.constant {name = "%c8_i32"} 8 : i32
    %c12_i32 = arith.constant {name = "%c12_i32"} 12 : i32
    %c30_i32 = arith.constant {name = "%c30_i32"} 30 : i32
    %c11_i32 = arith.constant {name = "%c11_i32"} 11 : i32
    %c10_i32 = arith.constant {name = "%c10_i32"} 10 : i32
    %c7_i32 = arith.constant {name = "%c7_i32"} 7 : i32
    %c6_i32 = arith.constant {name = "%c6_i32"} 6 : i32
    %c5_i32 = arith.constant {name = "%c5_i32"} 5 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c9_i32 = arith.constant {name = "%c9_i32"} 9 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c15_i32 = arith.constant {name = "%c15_i32"} 15 : i32
    %c20_i32 = arith.constant {name = "%c20_i32"} 20 : i32
    %c255_i32 = arith.constant {name = "%c255_i32"} 255 : i32
    %c24_i32 = arith.constant {name = "%c24_i32"} 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg4 = 0 to 16 {
      affine.store %c0_i32, %alloc[%arg4] : memref<16xi32>
    }
    affine.for %arg4 = 0 to 16 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      %alloc_1 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_1[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg4] {to = "prog"} : memref<16xi32>
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
    affine.for %arg4 = 0 to 4 {
      affine.store %c0_i32, %alloc_0[%arg4] : memref<4xi32>
    }
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 16 {
        %0 = affine.load %alloc[%arg5] {from = "prog"} : memref<16xi32>
        %alloc_1 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_1[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %2 = arith.shrsi %1, %c24_i32 : i32
        %3 = arith.andi %2, %c255_i32 : i32
        %alloc_2 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_2[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %5 = arith.shrsi %4, %c20_i32 : i32
        %6 = arith.andi %5, %c15_i32 : i32
        %alloc_3 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_3[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %8 = arith.shrsi %7, %c16_i32 : i32
        %9 = arith.andi %8, %c15_i32 : i32
        %alloc_4 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_4[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_1[] {from = "word2"} : memref<i32>
        %11 = arith.andi %10, %c65535_i32 : i32
        %alloc_5 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_5[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
        %13 = arith.cmpi eq, %12, %c9_i32 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
          %alloc_6 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_6[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_6[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          memref.store %22, %alloc_0[%16] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
          %15 = arith.cmpi eq, %14, %c1_i32 : i32
          scf.if %15 {
            %16 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
            %alloc_6 = memref.alloc() {name = "z2"} : memref<i32>
            affine.store %16, %alloc_6[] {to = "z2"} : memref<i32>
            %17 = affine.load %alloc_6[] {from = "z2"} : memref<i32>
            %18 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_0[%19] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
            %17 = arith.cmpi eq, %16, %c2_i32 : i32
            scf.if %17 {
              %18 = affine.load %alloc_4[] {from = "src"} : memref<i32>
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg0[%c3, %19] {from = "local_Bias"} : memref<4x2xi32, #map>
              %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
              %22 = arith.index_cast %21 : i32 to index
              memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
              %19 = arith.cmpi eq, %18, %c3_i32 : i32
              scf.if %19 {
                %20 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                %21 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                memref.store %20, %alloc_0[%22] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                %21 = arith.cmpi eq, %20, %c4_i32 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i33
                  %29 = arith.extsi %27 : i32 to i33
                  %30 = arith.addi %28, %29 : i33
                  %31 = arith.trunci %30 : i33 to i32
                  memref.store %31, %alloc_0[%23] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                  %23 = arith.cmpi eq, %22, %c5_i32 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.extsi %26 : i32 to i64
                    %31 = arith.extsi %29 : i32 to i64
                    %32 = arith.muli %30, %31 : i64
                    %33 = arith.trunci %32 : i64 to i32
                    memref.store %33, %alloc_0[%25] {to = "reg"} : memref<4xi32>
                  } else {
                    %24 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                    %25 = arith.cmpi eq, %24, %c6_i32 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                      %30 = arith.index_cast %29 : i32 to index
                      %31 = memref.load %alloc_0[%30] {from = "reg"} : memref<4xi32>
                      %32 = arith.cmpi sgt, %28, %31 : i32
                      scf.if %32 {
                        %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                        %34 = arith.index_cast %33 : i32 to index
                        %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                        %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %37 = arith.index_cast %36 : i32 to index
                        memref.store %35, %alloc_0[%37] {to = "reg"} : memref<4xi32>
                      }
                    } else {
                      %26 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                      %27 = arith.cmpi eq, %26, %c7_i32 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        %31 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                        %32 = arith.shrsi %30, %31 : i32
                        memref.store %32, %alloc_0[%29] {to = "reg"} : memref<4xi32>
                      } else {
                        %28 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                        %29 = arith.cmpi eq, %28, %c10_i32 : i32
                        scf.if %29 {
                          %30 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                          %31 = arith.index_cast %30 : i32 to index
                          %32 = memref.load %alloc_0[%31] {from = "reg"} : memref<4xi32>
                          %33 = affine.load %alloc_4[] {from = "src"} : memref<i32>
                          %34 = arith.index_cast %33 : i32 to index
                          %35 = memref.load %alloc_0[%34] {from = "reg"} : memref<4xi32>
                          %36 = arith.extsi %32 : i32 to i33
                          %37 = arith.extsi %35 : i32 to i33
                          %38 = arith.subi %36, %37 : i33
                          %39 = arith.trunci %38 : i33 to i32
                          memref.store %39, %alloc_0[%31] {to = "reg"} : memref<4xi32>
                        } else {
                          %30 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                          %31 = arith.cmpi eq, %30, %c11_i32 : i32
                          scf.if %31 {
                            %32 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %33 = arith.index_cast %32 : i32 to index
                            %34 = memref.load %alloc_0[%33] {from = "reg"} : memref<4xi32>
                            %alloc_6 = memref.alloc() {name = "e"} : memref<i32>
                            affine.store %34, %alloc_6[] {to = "e"} : memref<i32>
                            %35 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %36 = arith.cmpi slt, %35, %c0_i32 : i32
                            scf.if %36 {
                              affine.store %c0_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %37 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %38 = arith.cmpi sgt, %37, %c30_i32 : i32
                            scf.if %38 {
                              affine.store %c30_i32, %alloc_6[] {to = "e"} : memref<i32>
                            }
                            %39 = affine.load %alloc_6[] {from = "e"} : memref<i32>
                            %40 = arith.shli %c1_i32, %39 : i32
                            %41 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                            %42 = arith.index_cast %41 : i32 to index
                            memref.store %40, %alloc_0[%42] {to = "reg"} : memref<4xi32>
                          } else {
                            %32 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                            %33 = arith.cmpi eq, %32, %c12_i32 : i32
                            scf.if %33 {
                              %34 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                              %35 = arith.index_cast %34 : i32 to index
                              %36 = memref.load %alloc_0[%35] {from = "reg"} : memref<4xi32>
                              %alloc_6 = memref.alloc() {name = "d"} : memref<i32>
                              affine.store %36, %alloc_6[] {to = "d"} : memref<i32>
                              %37 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                              %38 = arith.cmpi sgt, %37, %c0_i32 : i32
                              scf.if %38 {
                                %39 = affine.load %alloc_5[] {from = "imm"} : memref<i32>
                                %40 = arith.shli %c1_i32, %39 : i32
                                %41 = affine.load %alloc_6[] {from = "d"} : memref<i32>
                                %42 = arith.floordivsi %40, %41 : i32
                                %43 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %44 = arith.index_cast %43 : i32 to index
                                memref.store %42, %alloc_0[%44] {to = "reg"} : memref<4xi32>
                              } else {
                                %39 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %40 = arith.index_cast %39 : i32 to index
                                memref.store %c0_i32, %alloc_0[%40] {to = "reg"} : memref<4xi32>
                              }
                            } else {
                              %34 = affine.load %alloc_2[] {from = "opcode"} : memref<i32>
                              %35 = arith.cmpi eq, %34, %c8_i32 : i32
                              scf.if %35 {
                                %36 = affine.load %alloc_3[] {from = "dst"} : memref<i32>
                                %37 = arith.index_cast %36 : i32 to index
                                %38 = memref.load %alloc_0[%37] {from = "reg"} : memref<4xi32>
                                allo.stream_put(%arg3, [], %38) : !allo.stream<i32, 2> contains i32
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } {loop_name = "pc2", op_name = "S_pc2_1", pipeline_ii = 1 : ui32}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_y_out_drain_0(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 0] {to = "local_Y"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_1(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 1] {to = "local_Y"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_2(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 2] {to = "local_Y"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain_3(%arg0: memref<4x4xi32, #map>, %arg1: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_", otypes = "", stypes = "_i"} {
    affine.for %arg2 = 0 to 4 {
      %0 = allo.stream_get(%arg1, []) : !allo.stream<i32, 2> -> i32
      affine.store %0, %arg0[%arg2, 3] {to = "local_Y"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<4x4xi8, #map>, %arg1: memref<4x4xi32, #map>, %arg2: memref<16xi32, #map1>, %arg3: memref<4x4x4xi8, #map2>, %arg4: memref<4x2xi32, #map>, %arg5: memref<4x4xi32, #map>) attributes {dataflow, itypes = "ssssss", top} {
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
    %16 = allo.stream_construct() {name = "mac_op_out_op_in_0_0"} : !allo.stream<i32, 2>
    %17 = allo.stream_construct() {name = "mac_op_out_op_in_0_1"} : !allo.stream<i32, 2>
    %18 = allo.stream_construct() {name = "mac_op_out_op_in_0_2"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "mac_op_out_op_in_0_3"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "mac_op_out_op_in_1_0"} : !allo.stream<i32, 2>
    %21 = allo.stream_construct() {name = "mac_op_out_op_in_1_1"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "mac_op_out_op_in_1_2"} : !allo.stream<i32, 2>
    %23 = allo.stream_construct() {name = "mac_op_out_op_in_1_3"} : !allo.stream<i32, 2>
    %24 = allo.stream_construct() {name = "mac_op_out_op_in_2_0"} : !allo.stream<i32, 2>
    %25 = allo.stream_construct() {name = "mac_op_out_op_in_2_1"} : !allo.stream<i32, 2>
    %26 = allo.stream_construct() {name = "mac_op_out_op_in_2_2"} : !allo.stream<i32, 2>
    %27 = allo.stream_construct() {name = "mac_op_out_op_in_2_3"} : !allo.stream<i32, 2>
    %28 = allo.stream_construct() {name = "mac_op_out_op_in_3_0"} : !allo.stream<i32, 2>
    %29 = allo.stream_construct() {name = "mac_op_out_op_in_3_1"} : !allo.stream<i32, 2>
    %30 = allo.stream_construct() {name = "mac_op_out_op_in_3_2"} : !allo.stream<i32, 2>
    %31 = allo.stream_construct() {name = "mac_op_out_op_in_3_3"} : !allo.stream<i32, 2>
    %32 = allo.stream_construct() {name = "mac_p_out_p_in_0_0"} : !allo.stream<i32, 2>
    %33 = allo.stream_construct() {name = "mac_p_out_p_in_0_1"} : !allo.stream<i32, 2>
    %34 = allo.stream_construct() {name = "mac_p_out_p_in_0_2"} : !allo.stream<i32, 2>
    %35 = allo.stream_construct() {name = "mac_p_out_p_in_0_3"} : !allo.stream<i32, 2>
    %36 = allo.stream_construct() {name = "mac_p_out_p_in_1_0"} : !allo.stream<i32, 2>
    %37 = allo.stream_construct() {name = "mac_p_out_p_in_1_1"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "mac_p_out_p_in_1_2"} : !allo.stream<i32, 2>
    %39 = allo.stream_construct() {name = "mac_p_out_p_in_1_3"} : !allo.stream<i32, 2>
    %40 = allo.stream_construct() {name = "mac_p_out_p_in_2_0"} : !allo.stream<i32, 2>
    %41 = allo.stream_construct() {name = "mac_p_out_p_in_2_1"} : !allo.stream<i32, 2>
    %42 = allo.stream_construct() {name = "mac_p_out_p_in_2_2"} : !allo.stream<i32, 2>
    %43 = allo.stream_construct() {name = "mac_p_out_p_in_2_3"} : !allo.stream<i32, 2>
    %44 = allo.stream_construct() {name = "mac_p_out_p_in_3_0"} : !allo.stream<i32, 2>
    %45 = allo.stream_construct() {name = "mac_p_out_p_in_3_1"} : !allo.stream<i32, 2>
    %46 = allo.stream_construct() {name = "mac_p_out_p_in_3_2"} : !allo.stream<i32, 2>
    %47 = allo.stream_construct() {name = "mac_p_out_p_in_3_3"} : !allo.stream<i32, 2>
    %48 = allo.stream_construct() {name = "vpu_op_out_op_in_0"} : !allo.stream<i32, 2>
    %49 = allo.stream_construct() {name = "vpu_op_out_op_in_1"} : !allo.stream<i32, 2>
    %50 = allo.stream_construct() {name = "vpu_op_out_op_in_2"} : !allo.stream<i32, 2>
    %51 = allo.stream_construct() {name = "vpu_op_out_op_in_3"} : !allo.stream<i32, 2>
    %52 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 2>
    %53 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 2>
    %54 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 2>
    %55 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 2>
    %56 = allo.stream_construct() {name = "mac_op_in_bind_0"} : !allo.stream<i32, 2>
    %57 = allo.stream_construct() {name = "mac_op_in_bind_1"} : !allo.stream<i32, 2>
    %58 = allo.stream_construct() {name = "mac_op_in_bind_2"} : !allo.stream<i32, 2>
    %59 = allo.stream_construct() {name = "mac_op_in_bind_3"} : !allo.stream<i32, 2>
    %60 = allo.stream_construct() {name = "vpu_z_in_bind_0"} : !allo.stream<i32, 2>
    %61 = allo.stream_construct() {name = "vpu_z_in_bind_1"} : !allo.stream<i32, 2>
    %62 = allo.stream_construct() {name = "vpu_z_in_bind_2"} : !allo.stream<i32, 2>
    %63 = allo.stream_construct() {name = "vpu_z_in_bind_3"} : !allo.stream<i32, 2>
    %64 = allo.stream_construct() {name = "vpu_op_in_bind_0"} : !allo.stream<i32, 2>
    %65 = allo.stream_construct() {name = "vpu_y_out_bind_0"} : !allo.stream<i32, 2>
    %66 = allo.stream_construct() {name = "vpu_y_out_bind_1"} : !allo.stream<i32, 2>
    %67 = allo.stream_construct() {name = "vpu_y_out_bind_2"} : !allo.stream<i32, 2>
    %68 = allo.stream_construct() {name = "vpu_y_out_bind_3"} : !allo.stream<i32, 2>
    call @mac_a_in_load_0(%arg0, %52) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_1(%arg0, %53) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_2(%arg0, %54) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_a_in_load_3(%arg0, %55) : (memref<4x4xi8, #map>, !allo.stream<i8, 2>) -> ()
    call @mac_op_in_load_0(%arg1, %56) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @mac_op_in_load_1(%arg1, %57) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @mac_op_in_load_2(%arg1, %58) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @mac_op_in_load_3(%arg1, %59) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_op_in_load_0(%arg2, %64) : (memref<16xi32, #map1>, !allo.stream<i32, 2>) -> ()
    call @mac_0_0(%arg3, %56, %17, %52, %1, %36) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_0_1(%arg3, %17, %18, %1, %2, %37) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_0_2(%arg3, %18, %19, %2, %3, %38) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_0_3(%arg3, %19, %3, %39) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_0(%arg3, %57, %21, %53, %36, %5, %40) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_1(%arg3, %21, %22, %5, %37, %6, %41) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_2(%arg3, %22, %23, %6, %38, %7, %42) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_1_3(%arg3, %23, %7, %39, %43) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_0(%arg3, %58, %25, %54, %40, %9, %44) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_1(%arg3, %25, %26, %9, %41, %10, %45) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_2(%arg3, %26, %27, %10, %42, %11, %46) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_2_3(%arg3, %27, %11, %43, %47) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_0(%arg3, %59, %29, %55, %44, %13, %60) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_1(%arg3, %29, %30, %13, %45, %14, %61) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_2(%arg3, %30, %31, %14, %46, %15, %62) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_3_3(%arg3, %31, %15, %47, %63) : (memref<4x4x4xi8, #map2>, !allo.stream<i32, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_0(%arg4, %64, %49, %60, %65) : (memref<4x2xi32, #map>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_1(%arg4, %49, %50, %61, %66) : (memref<4x2xi32, #map>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_2(%arg4, %50, %51, %62, %67) : (memref<4x2xi32, #map>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_3(%arg4, %51, %63, %68) : (memref<4x2xi32, #map>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_0(%arg5, %65) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_1(%arg5, %66) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_2(%arg5, %67) : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain_3(%arg5, %68) {last} : (memref<4x4xi32, #map>, !allo.stream<i32, 2>) -> ()
    return
  }
}
