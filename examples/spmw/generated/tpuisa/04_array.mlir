#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0, 0, 0)>
module {
  func.func @mac_a_in_load(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: !allo.stream<i8, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<4x4xi8, #map>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 4> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_op_in_load(%arg0: memref<5x4xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 5>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 5 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_MProg"} : memref<5x4xi32, #map>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 5> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_op_in_load(%arg0: memref<17xi32, #map1>, %arg1: index, %arg2: !allo.stream<i32, 17>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 17 {
      %0 = affine.load %arg0[%arg3] {from = "local_VProg"} : memref<17xi32, #map1>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 17> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r0(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg9 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
      %15 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %arg0[%arg1, %arg2, %16] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %18 = arith.extsi %17 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %18, %alloc_5[] {to = "wt"} : memref<i32>
      %19 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %20 = arith.cmpi eq, %19, %c1_i32 : i32
      scf.if %20 {
        %21 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %24 = arith.extsi %22 : i8 to i40
        %25 = arith.extsi %23 : i32 to i40
        %26 = arith.muli %24, %25 : i40
        %27 = arith.extsi %21 : i32 to i41
        %28 = arith.extsi %26 : i40 to i41
        %29 = arith.addi %27, %28 : i41
        allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i41
      } else {
        %21 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %22 = arith.cmpi eq, %21, %c2_i32 : i32
        scf.if %22 {
          %23 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %24 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %25 = arith.extsi %23 : i8 to i40
          %26 = arith.extsi %24 : i32 to i40
          %27 = arith.muli %25, %26 : i40
          allo.stream_put(%arg8, [], %27) : !allo.stream<i32, 2> contains i40
        } else {
          %23 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %23) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r1(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg9 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
      %15 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %arg0[%arg1, %arg2, %16] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %18 = arith.extsi %17 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %18, %alloc_5[] {to = "wt"} : memref<i32>
      %19 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %20 = arith.cmpi eq, %19, %c1_i32 : i32
      scf.if %20 {
        %21 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %24 = arith.extsi %22 : i8 to i40
        %25 = arith.extsi %23 : i32 to i40
        %26 = arith.muli %24, %25 : i40
        %27 = arith.extsi %21 : i32 to i41
        %28 = arith.extsi %26 : i40 to i41
        %29 = arith.addi %27, %28 : i41
        allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i41
      } else {
        %21 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %22 = arith.cmpi eq, %21, %c2_i32 : i32
        scf.if %22 {
          %23 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %24 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %25 = arith.extsi %23 : i8 to i40
          %26 = arith.extsi %24 : i32 to i40
          %27 = arith.muli %25, %26 : i40
          allo.stream_put(%arg8, [], %27) : !allo.stream<i32, 2> contains i40
        } else {
          %23 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %23) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r2(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_______", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_4[] {to = "p"} : memref<i32>
      %13 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %arg0[%arg1, %arg2, %15] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %17 = arith.extsi %16 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %17, %alloc_5[] {to = "wt"} : memref<i32>
      %18 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %19 = arith.cmpi eq, %18, %c1_i32 : i32
      scf.if %19 {
        %20 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %21 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %22 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %23 = arith.extsi %21 : i8 to i40
        %24 = arith.extsi %22 : i32 to i40
        %25 = arith.muli %23, %24 : i40
        %26 = arith.extsi %20 : i32 to i41
        %27 = arith.extsi %25 : i40 to i41
        %28 = arith.addi %26, %27 : i41
        allo.stream_put(%arg7, [], %28) : !allo.stream<i32, 2> contains i41
      } else {
        %20 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %21 = arith.cmpi eq, %20, %c2_i32 : i32
        scf.if %21 {
          %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %24 = arith.extsi %22 : i8 to i40
          %25 = arith.extsi %23 : i32 to i40
          %26 = arith.muli %24, %25 : i40
          allo.stream_put(%arg7, [], %26) : !allo.stream<i32, 2> contains i40
        } else {
          %22 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg7, [], %22) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r3(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg4, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %2 = arith.index_cast %1 {name = "%3"} : i32 to index
    scf.for %arg7 = %c0 to %2 step %c1 {
      %3 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "word"} : memref<i32>
      %4 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c24_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "opcode"} : memref<i32>
      %7 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %8 = arith.shrsi %7, %c16_i32 : i32
      %9 = arith.andi %8, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %9, %alloc_2[] {to = "tile"} : memref<i32>
      %10 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %10, %alloc_3[] {to = "a"} : memref<i8>
      %11 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %11, %alloc_4[] {to = "p"} : memref<i32>
      %12 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = memref.load %arg0[%arg1, %arg2, %13] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %15 = arith.extsi %14 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %15, %alloc_5[] {to = "wt"} : memref<i32>
      %16 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %17 = arith.cmpi eq, %16, %c1_i32 : i32
      scf.if %17 {
        %18 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %19 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %20 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %21 = arith.extsi %19 : i8 to i40
        %22 = arith.extsi %20 : i32 to i40
        %23 = arith.muli %21, %22 : i40
        %24 = arith.extsi %18 : i32 to i41
        %25 = arith.extsi %23 : i40 to i41
        %26 = arith.addi %24, %25 : i41
        allo.stream_put(%arg6, [], %26) : !allo.stream<i32, 2> contains i41
      } else {
        %18 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %19 = arith.cmpi eq, %18, %c2_i32 : i32
        scf.if %19 {
          %20 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %21 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %22 = arith.extsi %20 : i8 to i40
          %23 = arith.extsi %21 : i32 to i40
          %24 = arith.muli %22, %23 : i40
          allo.stream_put(%arg6, [], %24) : !allo.stream<i32, 2> contains i40
        } else {
          %20 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %20) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r4(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg9 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
      %15 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %arg0[%arg1, %arg2, %16] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %18 = arith.extsi %17 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %18, %alloc_5[] {to = "wt"} : memref<i32>
      %19 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %20 = arith.cmpi eq, %19, %c1_i32 : i32
      scf.if %20 {
        %21 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %24 = arith.extsi %22 : i8 to i40
        %25 = arith.extsi %23 : i32 to i40
        %26 = arith.muli %24, %25 : i40
        %27 = arith.extsi %21 : i32 to i41
        %28 = arith.extsi %26 : i40 to i41
        %29 = arith.addi %27, %28 : i41
        allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i41
      } else {
        %21 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %22 = arith.cmpi eq, %21, %c2_i32 : i32
        scf.if %22 {
          %23 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %24 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %25 = arith.extsi %23 : i8 to i40
          %26 = arith.extsi %24 : i32 to i40
          %27 = arith.muli %25, %26 : i40
          allo.stream_put(%arg8, [], %27) : !allo.stream<i32, 2> contains i40
        } else {
          %23 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %23) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r5(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg4, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %2 = arith.index_cast %1 {name = "%3"} : i32 to index
    scf.for %arg7 = %c0 to %2 step %c1 {
      %3 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "word"} : memref<i32>
      %4 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c24_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "opcode"} : memref<i32>
      %7 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %8 = arith.shrsi %7, %c16_i32 : i32
      %9 = arith.andi %8, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %9, %alloc_2[] {to = "tile"} : memref<i32>
      %10 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %10, %alloc_3[] {to = "a"} : memref<i8>
      %11 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %11, %alloc_4[] {to = "p"} : memref<i32>
      %12 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = memref.load %arg0[%arg1, %arg2, %13] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %15 = arith.extsi %14 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %15, %alloc_5[] {to = "wt"} : memref<i32>
      %16 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %17 = arith.cmpi eq, %16, %c1_i32 : i32
      scf.if %17 {
        %18 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %19 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %20 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %21 = arith.extsi %19 : i8 to i40
        %22 = arith.extsi %20 : i32 to i40
        %23 = arith.muli %21, %22 : i40
        %24 = arith.extsi %18 : i32 to i41
        %25 = arith.extsi %23 : i40 to i41
        %26 = arith.addi %24, %25 : i41
        allo.stream_put(%arg6, [], %26) : !allo.stream<i32, 2> contains i41
      } else {
        %18 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %19 = arith.cmpi eq, %18, %c2_i32 : i32
        scf.if %19 {
          %20 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %21 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %22 = arith.extsi %20 : i8 to i40
          %23 = arith.extsi %21 : i32 to i40
          %24 = arith.muli %22, %23 : i40
          allo.stream_put(%arg6, [], %24) : !allo.stream<i32, 2> contains i40
        } else {
          %20 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %20) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r6(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %0 = allo.stream_get(%arg4, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %2 = arith.index_cast %1 {name = "%3"} : i32 to index
    scf.for %arg6 = %c0 to %2 step %c1 {
      %3 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %3, %alloc_0[] {to = "word"} : memref<i32>
      %4 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %5 = arith.shrsi %4, %c24_i32 : i32
      %6 = arith.andi %5, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %6, %alloc_1[] {to = "opcode"} : memref<i32>
      %7 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %8 = arith.shrsi %7, %c16_i32 : i32
      %9 = arith.andi %8, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %9, %alloc_2[] {to = "tile"} : memref<i32>
      %10 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %10, %alloc_3[] {to = "a"} : memref<i8>
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_4[] {to = "p"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %arg0[%arg1, %arg2, %12] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %14 = arith.extsi %13 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %14, %alloc_5[] {to = "wt"} : memref<i32>
      %15 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %16 = arith.cmpi eq, %15, %c1_i32 : i32
      scf.if %16 {
        %17 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %18 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %19 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %20 = arith.extsi %18 : i8 to i40
        %21 = arith.extsi %19 : i32 to i40
        %22 = arith.muli %20, %21 : i40
        %23 = arith.extsi %17 : i32 to i41
        %24 = arith.extsi %22 : i40 to i41
        %25 = arith.addi %23, %24 : i41
        allo.stream_put(%arg5, [], %25) : !allo.stream<i32, 2> contains i41
      } else {
        %17 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %18 = arith.cmpi eq, %17, %c2_i32 : i32
        scf.if %18 {
          %19 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %20 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %21 = arith.extsi %19 : i8 to i40
          %22 = arith.extsi %20 : i32 to i40
          %23 = arith.muli %21, %22 : i40
          allo.stream_put(%arg5, [], %23) : !allo.stream<i32, 2> contains i40
        } else {
          %19 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg5, [], %19) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r7(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg9 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_4[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %14) : !allo.stream<i8, 2> contains i8
      %15 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %arg0[%arg1, %arg2, %16] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %18 = arith.extsi %17 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %18, %alloc_5[] {to = "wt"} : memref<i32>
      %19 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %20 = arith.cmpi eq, %19, %c1_i32 : i32
      scf.if %20 {
        %21 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %24 = arith.extsi %22 : i8 to i40
        %25 = arith.extsi %23 : i32 to i40
        %26 = arith.muli %24, %25 : i40
        %27 = arith.extsi %21 : i32 to i41
        %28 = arith.extsi %26 : i40 to i41
        %29 = arith.addi %27, %28 : i41
        allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i41
      } else {
        %21 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %22 = arith.cmpi eq, %21, %c2_i32 : i32
        scf.if %22 {
          %23 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %24 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %25 = arith.extsi %23 : i8 to i40
          %26 = arith.extsi %24 : i32 to i40
          %27 = arith.muli %25, %26 : i40
          allo.stream_put(%arg8, [], %27) : !allo.stream<i32, 2> contains i40
        } else {
          %23 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %23) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r8(%arg0: memref<4x4x4xi8, #map2>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_______", otypes = ""} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %0 = allo.stream_get(%arg5, []) {name = "%0"} : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %3 = arith.index_cast %2 {name = "%4"} : i32 to index
    scf.for %arg8 = %c0 to %3 step %c1 {
      %4 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %4, %alloc_0[] {to = "word"} : memref<i32>
      %5 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %5) : !allo.stream<i32, 2> contains i32
      %6 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %7 = arith.shrsi %6, %c24_i32 : i32
      %8 = arith.andi %7, %c255_i32 : i32
      %alloc_1 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_1[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      %10 = arith.shrsi %9, %c16_i32 : i32
      %11 = arith.andi %10, %c255_i32 : i32
      %alloc_2 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_2[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_3 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_3[] {to = "a"} : memref<i8>
      %alloc_4 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_4[] {to = "p"} : memref<i32>
      %13 = affine.load %alloc_3[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      %14 = affine.load %alloc_2[] {from = "tile"} : memref<i32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %arg0[%arg1, %arg2, %15] {from = "local_W"} : memref<4x4x4xi8, #map2>
      %17 = arith.extsi %16 : i8 to i32
      %alloc_5 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %17, %alloc_5[] {to = "wt"} : memref<i32>
      %18 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
      %19 = arith.cmpi eq, %18, %c1_i32 : i32
      scf.if %19 {
        %20 = affine.load %alloc_4[] {from = "p"} : memref<i32>
        %21 = affine.load %alloc_3[] {from = "a"} : memref<i8>
        %22 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
        %23 = arith.extsi %21 : i8 to i40
        %24 = arith.extsi %22 : i32 to i40
        %25 = arith.muli %23, %24 : i40
        %26 = arith.extsi %20 : i32 to i41
        %27 = arith.extsi %25 : i40 to i41
        %28 = arith.addi %26, %27 : i41
        allo.stream_put(%arg7, [], %28) : !allo.stream<i32, 2> contains i41
      } else {
        %20 = affine.load %alloc_1[] {from = "opcode"} : memref<i32>
        %21 = arith.cmpi eq, %20, %c2_i32 : i32
        scf.if %21 {
          %22 = affine.load %alloc_3[] {from = "a"} : memref<i8>
          %23 = affine.load %alloc_5[] {from = "wt"} : memref<i32>
          %24 = arith.extsi %22 : i8 to i40
          %25 = arith.extsi %23 : i32 to i40
          %26 = arith.muli %24, %25 : i40
          allo.stream_put(%arg7, [], %26) : !allo.stream<i32, 2> contains i40
        } else {
          %22 = affine.load %alloc_4[] {from = "p"} : memref<i32>
          allo.stream_put(%arg7, [], %22) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_r0(%arg0: memref<4x2xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 4>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c1_i33 = arith.constant 1 : i33
    %c16384_i32 = arith.constant 16384 : i32
    %c16_i33 = arith.constant 16 : i33
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c12_i32 = arith.constant 12 : i32
    %c30_i32 = arith.constant 30 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c9_i32 = arith.constant 9 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %0 = allo.stream_get(%arg2, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "header"} : memref<i32>
    %3 = arith.andi %2, %c65535_i32 {name = "%3"} : i32
    %alloc_0 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %3, %alloc_0[] {to = "plen"} : memref<i32>
    %4 = affine.load %alloc[] {from = "header"} : memref<i32>
    %5 = arith.shrsi %4, %c16_i32 {name = "%5"} : i32
    %6 = arith.andi %5, %c65535_i32 {name = "%6"} : i32
    %alloc_1 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %6, %alloc_1[] {to = "nouts"} : memref<i32>
    %alloc_2 = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg6 = 0 to 16 {
      affine.store %c0_i32, %alloc_2[%arg6] : memref<16xi32>
    }
    %7 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %8 = arith.index_cast %7 {name = "%9"} : i32 to index
    scf.for %arg6 = %c0 to %8 step %c1 {
      %22 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_10 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %22, %alloc_10[] {to = "word"} : memref<i32>
      %23 = affine.load %alloc_10[] {from = "word"} : memref<i32>
      memref.store %23, %alloc_2[%arg6] {to = "prog"} : memref<16xi32>
      %24 = affine.load %alloc_10[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %24) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %9 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %10 = arith.extsi %9 {name = "%13"} : i32 to i33
    %11 = arith.subi %c16_i33, %10 {name = "%14"} : i33
    %12 = arith.index_cast %11 {name = "%15"} : i33 to index
    scf.for %arg6 = %c0 to %12 step %c1 {
      %22 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_10 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %22, %alloc_10[] {to = "spare"} : memref<i32>
      %23 = affine.load %alloc_10[] {from = "spare"} : memref<i32>
      allo.stream_put(%arg3, [], %23) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_pad", op_name = "S__pad_1", pipeline_ii = 1 : ui32}
    %13 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32, #map>
    %alloc_3 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %13, %alloc_3[] {to = "denom"} : memref<i32>
    %alloc_4 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32, %alloc_4[] {to = "rcp"} : memref<i32>
    %14 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
    %15 = arith.cmpi sgt, %14, %c0_i32 {name = "%18"} : i32
    scf.if %15 {
      %22 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
      %23 = arith.floordivsi %c16384_i32, %22 : i32
      affine.store %23, %alloc_4[] {to = "rcp"} : memref<i32>
    }
    %alloc_5 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32, %alloc_5[] {to = "r0"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32, %alloc_6[] {to = "r1"} : memref<i32>
    %alloc_7 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32, %alloc_7[] {to = "r2"} : memref<i32>
    %alloc_8 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32, %alloc_8[] {to = "r3"} : memref<i32>
    %alloc_9 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
    %16 = affine.load %alloc_1[] {from = "nouts"} : memref<i32>
    %17 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %18 = arith.extsi %16 {name = "%21"} : i32 to i64
    %19 = arith.extsi %17 {name = "%22"} : i32 to i64
    %20 = arith.muli %18, %19 {name = "%23"} : i64
    %21 = arith.index_cast %20 {name = "%24"} : i64 to index
    scf.for %arg6 = %c0 to %21 step %c1 {
      %22 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %alloc_2[%23] {from = "prog"} : memref<16xi32>
      %alloc_10 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %24, %alloc_10[] {to = "word2"} : memref<i32>
      %25 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %26 = arith.shrsi %25, %c24_i32 : i32
      %27 = arith.andi %26, %c255_i32 : i32
      %alloc_11 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %27, %alloc_11[] {to = "opcode"} : memref<i32>
      %28 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %29 = arith.shrsi %28, %c20_i32 : i32
      %30 = arith.andi %29, %c15_i32 : i32
      %alloc_12 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %30, %alloc_12[] {to = "dst"} : memref<i32>
      %31 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %32 = arith.shrsi %31, %c16_i32 : i32
      %33 = arith.andi %32, %c15_i32 : i32
      %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %33, %alloc_13[] {to = "src"} : memref<i32>
      %34 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %35 = arith.andi %34, %c65535_i32 : i32
      %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %35, %alloc_14[] {to = "imm"} : memref<i32>
      %36 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_15 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %36, %alloc_15[] {to = "d"} : memref<i32>
      %37 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
      %38 = arith.cmpi eq, %37, %c1_i32 : i32
      scf.if %38 {
        %53 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %53, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %53 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c2_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %55, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %55 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c3_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %57, %alloc_15[] {to = "d"} : memref<i32>
          }
        }
      }
      %39 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_16 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %39, %alloc_16[] {to = "a"} : memref<i32>
      %40 = affine.load %alloc_13[] {from = "src"} : memref<i32>
      %41 = arith.cmpi eq, %40, %c1_i32 : i32
      scf.if %41 {
        %53 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %53, %alloc_16[] {to = "a"} : memref<i32>
      } else {
        %53 = affine.load %alloc_13[] {from = "src"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c2_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %55, %alloc_16[] {to = "a"} : memref<i32>
        } else {
          %55 = affine.load %alloc_13[] {from = "src"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c3_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %57, %alloc_16[] {to = "a"} : memref<i32>
          }
        }
      }
      %alloc_17 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32, %alloc_17[] {to = "wr"} : memref<i32>
      %42 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
      %43 = arith.cmpi eq, %42, %c9_i32 : i32
      scf.if %43 {
        %53 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_18 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %53, %alloc_18[] {to = "zz"} : memref<i32>
        %54 = affine.load %alloc_15[] {from = "d"} : memref<i32>
        %55 = affine.load %alloc_18[] {from = "zz"} : memref<i32>
        %56 = arith.extsi %54 : i32 to i33
        %57 = arith.extsi %55 : i32 to i33
        %58 = arith.addi %56, %57 : i33
        %59 = arith.trunci %58 : i33 to i32
        affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %53 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c1_i32 : i32
        scf.if %54 {
          %55 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_18 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %55, %alloc_18[] {to = "z2"} : memref<i32>
          %56 = affine.load %alloc_18[] {from = "z2"} : memref<i32>
          affine.store %56, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %55 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c2_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_13[] {from = "src"} : memref<i32>
            %58 = arith.index_cast %57 : i32 to index
            %59 = memref.load %arg0[%arg1, %58] {from = "local_Bias"} : memref<4x2xi32, #map>
            affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
          } else {
            %57 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
            %58 = arith.cmpi eq, %57, %c3_i32 : i32
            scf.if %58 {
              %59 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
            } else {
              %59 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
              %60 = arith.cmpi eq, %59, %c4_i32 : i32
              scf.if %60 {
                %61 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                %62 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                %63 = arith.extsi %61 : i32 to i33
                %64 = arith.extsi %62 : i32 to i33
                %65 = arith.addi %63, %64 : i33
                %66 = arith.trunci %65 : i33 to i32
                affine.store %66, %alloc_15[] {to = "d"} : memref<i32>
              } else {
                %61 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                %62 = arith.cmpi eq, %61, %c5_i32 : i32
                scf.if %62 {
                  %63 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                  %64 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                  %65 = arith.extsi %63 : i32 to i64
                  %66 = arith.extsi %64 : i32 to i64
                  %67 = arith.muli %65, %66 : i64
                  %68 = arith.trunci %67 : i64 to i32
                  affine.store %68, %alloc_15[] {to = "d"} : memref<i32>
                } else {
                  %63 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                  %64 = arith.cmpi eq, %63, %c6_i32 : i32
                  scf.if %64 {
                    %65 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                    %66 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                    %67 = arith.cmpi sgt, %65, %66 : i32
                    scf.if %67 {
                      %68 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                      affine.store %68, %alloc_15[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %65 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                    %66 = arith.cmpi eq, %65, %c7_i32 : i32
                    scf.if %66 {
                      %67 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                      %68 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %69 = arith.shrsi %67, %68 : i32
                      affine.store %69, %alloc_15[] {to = "d"} : memref<i32>
                    } else {
                      %67 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                      %68 = arith.cmpi eq, %67, %c10_i32 : i32
                      scf.if %68 {
                        %69 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                        %70 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                        %71 = arith.extsi %69 : i32 to i33
                        %72 = arith.extsi %70 : i32 to i33
                        %73 = arith.subi %71, %72 : i33
                        %74 = arith.trunci %73 : i33 to i32
                        affine.store %74, %alloc_15[] {to = "d"} : memref<i32>
                      } else {
                        %69 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                        %70 = arith.cmpi eq, %69, %c11_i32 : i32
                        scf.if %70 {
                          %71 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                          %alloc_18 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %71, %alloc_18[] {to = "e"} : memref<i32>
                          %72 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %73 = arith.cmpi slt, %72, %c0_i32 : i32
                          scf.if %73 {
                            affine.store %c0_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %74 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %75 = arith.cmpi sgt, %74, %c30_i32 : i32
                          scf.if %75 {
                            affine.store %c30_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %76 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %77 = arith.shli %c1_i32, %76 : i32
                          affine.store %77, %alloc_15[] {to = "d"} : memref<i32>
                        } else {
                          %71 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                          %72 = arith.cmpi eq, %71, %c12_i32 : i32
                          scf.if %72 {
                            %73 = affine.load %alloc_4[] {from = "rcp"} : memref<i32>
                            affine.store %73, %alloc_15[] {to = "d"} : memref<i32>
                          } else {
                            %73 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                            %74 = arith.cmpi eq, %73, %c8_i32 : i32
                            scf.if %74 {
                              %75 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg4, [], %75) : !allo.stream<i32, 4> contains i32
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
                            } else {
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
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
      %44 = affine.load %alloc_17[] {from = "wr"} : memref<i32>
      %45 = arith.cmpi eq, %44, %c1_i32 : i32
      scf.if %45 {
        %53 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c0_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_15[] {from = "d"} : memref<i32>
          affine.store %55, %alloc_5[] {to = "r0"} : memref<i32>
        } else {
          %55 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c1_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_15[] {from = "d"} : memref<i32>
            affine.store %57, %alloc_6[] {to = "r1"} : memref<i32>
          } else {
            %57 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
            %58 = arith.cmpi eq, %57, %c2_i32 : i32
            scf.if %58 {
              %59 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %59, %alloc_7[] {to = "r2"} : memref<i32>
            } else {
              %59 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %59, %alloc_8[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %46 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %47 = arith.extsi %46 : i32 to i33
      %48 = arith.addi %47, %c1_i33 : i33
      %49 = arith.trunci %48 : i33 to i32
      affine.store %49, %alloc_9[] {to = "pc2"} : memref<i32>
      %50 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %51 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
      %52 = arith.cmpi eq, %50, %51 : i32
      scf.if %52 {
        affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_r1(%arg0: memref<4x2xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 4>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %c1_i33 = arith.constant 1 : i33
    %c16384_i32 = arith.constant 16384 : i32
    %c16_i33 = arith.constant 16 : i33
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c12_i32 = arith.constant 12 : i32
    %c30_i32 = arith.constant 30 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c9_i32 = arith.constant 9 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %0 = allo.stream_get(%arg2, []) {name = "%0"} : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    %2 = arith.andi %1, %c65535_i32 {name = "%2"} : i32
    %alloc_0 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %2, %alloc_0[] {to = "plen"} : memref<i32>
    %3 = affine.load %alloc[] {from = "header"} : memref<i32>
    %4 = arith.shrsi %3, %c16_i32 {name = "%4"} : i32
    %5 = arith.andi %4, %c65535_i32 {name = "%5"} : i32
    %alloc_1 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %5, %alloc_1[] {to = "nouts"} : memref<i32>
    %alloc_2 = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg5 = 0 to 16 {
      affine.store %c0_i32, %alloc_2[%arg5] : memref<16xi32>
    }
    %6 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %7 = arith.index_cast %6 {name = "%8"} : i32 to index
    scf.for %arg5 = %c0 to %7 step %c1 {
      %21 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_10 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %21, %alloc_10[] {to = "word"} : memref<i32>
      %22 = affine.load %alloc_10[] {from = "word"} : memref<i32>
      memref.store %22, %alloc_2[%arg5] {to = "prog"} : memref<16xi32>
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %8 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %9 = arith.extsi %8 {name = "%12"} : i32 to i33
    %10 = arith.subi %c16_i33, %9 {name = "%13"} : i33
    %11 = arith.index_cast %10 {name = "%14"} : i33 to index
    scf.for %arg5 = %c0 to %11 step %c1 {
      %21 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_10 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %21, %alloc_10[] {to = "spare"} : memref<i32>
    } {loop_name = "_pad", op_name = "S__pad_1", pipeline_ii = 1 : ui32}
    %12 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32, #map>
    %alloc_3 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %12, %alloc_3[] {to = "denom"} : memref<i32>
    %alloc_4 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32, %alloc_4[] {to = "rcp"} : memref<i32>
    %13 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
    %14 = arith.cmpi sgt, %13, %c0_i32 {name = "%17"} : i32
    scf.if %14 {
      %21 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
      %22 = arith.floordivsi %c16384_i32, %21 : i32
      affine.store %22, %alloc_4[] {to = "rcp"} : memref<i32>
    }
    %alloc_5 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32, %alloc_5[] {to = "r0"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32, %alloc_6[] {to = "r1"} : memref<i32>
    %alloc_7 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32, %alloc_7[] {to = "r2"} : memref<i32>
    %alloc_8 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32, %alloc_8[] {to = "r3"} : memref<i32>
    %alloc_9 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
    %15 = affine.load %alloc_1[] {from = "nouts"} : memref<i32>
    %16 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %17 = arith.extsi %15 {name = "%20"} : i32 to i64
    %18 = arith.extsi %16 {name = "%21"} : i32 to i64
    %19 = arith.muli %17, %18 {name = "%22"} : i64
    %20 = arith.index_cast %19 {name = "%23"} : i64 to index
    scf.for %arg5 = %c0 to %20 step %c1 {
      %21 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %22 = arith.index_cast %21 : i32 to index
      %23 = memref.load %alloc_2[%22] {from = "prog"} : memref<16xi32>
      %alloc_10 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %23, %alloc_10[] {to = "word2"} : memref<i32>
      %24 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %25 = arith.shrsi %24, %c24_i32 : i32
      %26 = arith.andi %25, %c255_i32 : i32
      %alloc_11 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %26, %alloc_11[] {to = "opcode"} : memref<i32>
      %27 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %28 = arith.shrsi %27, %c20_i32 : i32
      %29 = arith.andi %28, %c15_i32 : i32
      %alloc_12 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %29, %alloc_12[] {to = "dst"} : memref<i32>
      %30 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %31 = arith.shrsi %30, %c16_i32 : i32
      %32 = arith.andi %31, %c15_i32 : i32
      %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %32, %alloc_13[] {to = "src"} : memref<i32>
      %33 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %34 = arith.andi %33, %c65535_i32 : i32
      %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %34, %alloc_14[] {to = "imm"} : memref<i32>
      %35 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_15 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %35, %alloc_15[] {to = "d"} : memref<i32>
      %36 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
      %37 = arith.cmpi eq, %36, %c1_i32 : i32
      scf.if %37 {
        %52 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %52, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %52 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %53 = arith.cmpi eq, %52, %c2_i32 : i32
        scf.if %53 {
          %54 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %54, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %54 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %55 = arith.cmpi eq, %54, %c3_i32 : i32
          scf.if %55 {
            %56 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %56, %alloc_15[] {to = "d"} : memref<i32>
          }
        }
      }
      %38 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_16 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %38, %alloc_16[] {to = "a"} : memref<i32>
      %39 = affine.load %alloc_13[] {from = "src"} : memref<i32>
      %40 = arith.cmpi eq, %39, %c1_i32 : i32
      scf.if %40 {
        %52 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %52, %alloc_16[] {to = "a"} : memref<i32>
      } else {
        %52 = affine.load %alloc_13[] {from = "src"} : memref<i32>
        %53 = arith.cmpi eq, %52, %c2_i32 : i32
        scf.if %53 {
          %54 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %54, %alloc_16[] {to = "a"} : memref<i32>
        } else {
          %54 = affine.load %alloc_13[] {from = "src"} : memref<i32>
          %55 = arith.cmpi eq, %54, %c3_i32 : i32
          scf.if %55 {
            %56 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %56, %alloc_16[] {to = "a"} : memref<i32>
          }
        }
      }
      %alloc_17 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32, %alloc_17[] {to = "wr"} : memref<i32>
      %41 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
      %42 = arith.cmpi eq, %41, %c9_i32 : i32
      scf.if %42 {
        %52 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
        %alloc_18 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %52, %alloc_18[] {to = "zz"} : memref<i32>
        %53 = affine.load %alloc_15[] {from = "d"} : memref<i32>
        %54 = affine.load %alloc_18[] {from = "zz"} : memref<i32>
        %55 = arith.extsi %53 : i32 to i33
        %56 = arith.extsi %54 : i32 to i33
        %57 = arith.addi %55, %56 : i33
        %58 = arith.trunci %57 : i33 to i32
        affine.store %58, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %52 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
        %53 = arith.cmpi eq, %52, %c1_i32 : i32
        scf.if %53 {
          %54 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
          %alloc_18 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %54, %alloc_18[] {to = "z2"} : memref<i32>
          %55 = affine.load %alloc_18[] {from = "z2"} : memref<i32>
          affine.store %55, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %54 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
          %55 = arith.cmpi eq, %54, %c2_i32 : i32
          scf.if %55 {
            %56 = affine.load %alloc_13[] {from = "src"} : memref<i32>
            %57 = arith.index_cast %56 : i32 to index
            %58 = memref.load %arg0[%arg1, %57] {from = "local_Bias"} : memref<4x2xi32, #map>
            affine.store %58, %alloc_15[] {to = "d"} : memref<i32>
          } else {
            %56 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
            %57 = arith.cmpi eq, %56, %c3_i32 : i32
            scf.if %57 {
              %58 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              affine.store %58, %alloc_15[] {to = "d"} : memref<i32>
            } else {
              %58 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
              %59 = arith.cmpi eq, %58, %c4_i32 : i32
              scf.if %59 {
                %60 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                %61 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                %62 = arith.extsi %60 : i32 to i33
                %63 = arith.extsi %61 : i32 to i33
                %64 = arith.addi %62, %63 : i33
                %65 = arith.trunci %64 : i33 to i32
                affine.store %65, %alloc_15[] {to = "d"} : memref<i32>
              } else {
                %60 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                %61 = arith.cmpi eq, %60, %c5_i32 : i32
                scf.if %61 {
                  %62 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                  %63 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                  %64 = arith.extsi %62 : i32 to i64
                  %65 = arith.extsi %63 : i32 to i64
                  %66 = arith.muli %64, %65 : i64
                  %67 = arith.trunci %66 : i64 to i32
                  affine.store %67, %alloc_15[] {to = "d"} : memref<i32>
                } else {
                  %62 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                  %63 = arith.cmpi eq, %62, %c6_i32 : i32
                  scf.if %63 {
                    %64 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                    %65 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                    %66 = arith.cmpi sgt, %64, %65 : i32
                    scf.if %66 {
                      %67 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                      affine.store %67, %alloc_15[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %64 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                    %65 = arith.cmpi eq, %64, %c7_i32 : i32
                    scf.if %65 {
                      %66 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                      %67 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %68 = arith.shrsi %66, %67 : i32
                      affine.store %68, %alloc_15[] {to = "d"} : memref<i32>
                    } else {
                      %66 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                      %67 = arith.cmpi eq, %66, %c10_i32 : i32
                      scf.if %67 {
                        %68 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                        %69 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                        %70 = arith.extsi %68 : i32 to i33
                        %71 = arith.extsi %69 : i32 to i33
                        %72 = arith.subi %70, %71 : i33
                        %73 = arith.trunci %72 : i33 to i32
                        affine.store %73, %alloc_15[] {to = "d"} : memref<i32>
                      } else {
                        %68 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                        %69 = arith.cmpi eq, %68, %c11_i32 : i32
                        scf.if %69 {
                          %70 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                          %alloc_18 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %70, %alloc_18[] {to = "e"} : memref<i32>
                          %71 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %72 = arith.cmpi slt, %71, %c0_i32 : i32
                          scf.if %72 {
                            affine.store %c0_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %73 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %74 = arith.cmpi sgt, %73, %c30_i32 : i32
                          scf.if %74 {
                            affine.store %c30_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %75 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %76 = arith.shli %c1_i32, %75 : i32
                          affine.store %76, %alloc_15[] {to = "d"} : memref<i32>
                        } else {
                          %70 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                          %71 = arith.cmpi eq, %70, %c12_i32 : i32
                          scf.if %71 {
                            %72 = affine.load %alloc_4[] {from = "rcp"} : memref<i32>
                            affine.store %72, %alloc_15[] {to = "d"} : memref<i32>
                          } else {
                            %72 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                            %73 = arith.cmpi eq, %72, %c8_i32 : i32
                            scf.if %73 {
                              %74 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg3, [], %74) : !allo.stream<i32, 4> contains i32
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
                            } else {
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
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
      %43 = affine.load %alloc_17[] {from = "wr"} : memref<i32>
      %44 = arith.cmpi eq, %43, %c1_i32 : i32
      scf.if %44 {
        %52 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %53 = arith.cmpi eq, %52, %c0_i32 : i32
        scf.if %53 {
          %54 = affine.load %alloc_15[] {from = "d"} : memref<i32>
          affine.store %54, %alloc_5[] {to = "r0"} : memref<i32>
        } else {
          %54 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %55 = arith.cmpi eq, %54, %c1_i32 : i32
          scf.if %55 {
            %56 = affine.load %alloc_15[] {from = "d"} : memref<i32>
            affine.store %56, %alloc_6[] {to = "r1"} : memref<i32>
          } else {
            %56 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
            %57 = arith.cmpi eq, %56, %c2_i32 : i32
            scf.if %57 {
              %58 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %58, %alloc_7[] {to = "r2"} : memref<i32>
            } else {
              %58 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %58, %alloc_8[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %45 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %46 = arith.extsi %45 : i32 to i33
      %47 = arith.addi %46, %c1_i33 : i33
      %48 = arith.trunci %47 : i33 to i32
      affine.store %48, %alloc_9[] {to = "pc2"} : memref<i32>
      %49 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %50 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
      %51 = arith.cmpi eq, %49, %50 : i32
      scf.if %51 {
        affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_r2(%arg0: memref<4x2xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 17>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 4>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c1_i33 = arith.constant 1 : i33
    %c16384_i32 = arith.constant 16384 : i32
    %c16_i33 = arith.constant 16 : i33
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c12_i32 = arith.constant 12 : i32
    %c30_i32 = arith.constant 30 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c9_i32 = arith.constant 9 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c16_i32 = arith.constant {name = "%c16_i32"} 16 : i32
    %c65535_i32 = arith.constant {name = "%c65535_i32"} 65535 : i32
    %0 = allo.stream_get(%arg2, []) {name = "%0"} : !allo.stream<i32, 17> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "header"} : memref<i32>
    %3 = arith.andi %2, %c65535_i32 {name = "%3"} : i32
    %alloc_0 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %3, %alloc_0[] {to = "plen"} : memref<i32>
    %4 = affine.load %alloc[] {from = "header"} : memref<i32>
    %5 = arith.shrsi %4, %c16_i32 {name = "%5"} : i32
    %6 = arith.andi %5, %c65535_i32 {name = "%6"} : i32
    %alloc_1 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %6, %alloc_1[] {to = "nouts"} : memref<i32>
    %alloc_2 = memref.alloc() {name = "prog"} : memref<16xi32>
    affine.for %arg6 = 0 to 16 {
      affine.store %c0_i32, %alloc_2[%arg6] : memref<16xi32>
    }
    %7 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %8 = arith.index_cast %7 {name = "%9"} : i32 to index
    scf.for %arg6 = %c0 to %8 step %c1 {
      %22 = allo.stream_get(%arg2, []) : !allo.stream<i32, 17> -> i32
      %alloc_10 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %22, %alloc_10[] {to = "word"} : memref<i32>
      %23 = affine.load %alloc_10[] {from = "word"} : memref<i32>
      memref.store %23, %alloc_2[%arg6] {to = "prog"} : memref<16xi32>
      %24 = affine.load %alloc_10[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %24) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    %9 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %10 = arith.extsi %9 {name = "%13"} : i32 to i33
    %11 = arith.subi %c16_i33, %10 {name = "%14"} : i33
    %12 = arith.index_cast %11 {name = "%15"} : i33 to index
    scf.for %arg6 = %c0 to %12 step %c1 {
      %22 = allo.stream_get(%arg2, []) : !allo.stream<i32, 17> -> i32
      %alloc_10 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %22, %alloc_10[] {to = "spare"} : memref<i32>
      %23 = affine.load %alloc_10[] {from = "spare"} : memref<i32>
      allo.stream_put(%arg3, [], %23) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_pad", op_name = "S__pad_1", pipeline_ii = 1 : ui32}
    %13 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32, #map>
    %alloc_3 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %13, %alloc_3[] {to = "denom"} : memref<i32>
    %alloc_4 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32, %alloc_4[] {to = "rcp"} : memref<i32>
    %14 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
    %15 = arith.cmpi sgt, %14, %c0_i32 {name = "%18"} : i32
    scf.if %15 {
      %22 = affine.load %alloc_3[] {from = "denom"} : memref<i32>
      %23 = arith.floordivsi %c16384_i32, %22 : i32
      affine.store %23, %alloc_4[] {to = "rcp"} : memref<i32>
    }
    %alloc_5 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32, %alloc_5[] {to = "r0"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32, %alloc_6[] {to = "r1"} : memref<i32>
    %alloc_7 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32, %alloc_7[] {to = "r2"} : memref<i32>
    %alloc_8 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32, %alloc_8[] {to = "r3"} : memref<i32>
    %alloc_9 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
    %16 = affine.load %alloc_1[] {from = "nouts"} : memref<i32>
    %17 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
    %18 = arith.extsi %16 {name = "%21"} : i32 to i64
    %19 = arith.extsi %17 {name = "%22"} : i32 to i64
    %20 = arith.muli %18, %19 {name = "%23"} : i64
    %21 = arith.index_cast %20 {name = "%24"} : i64 to index
    scf.for %arg6 = %c0 to %21 step %c1 {
      %22 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %alloc_2[%23] {from = "prog"} : memref<16xi32>
      %alloc_10 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %24, %alloc_10[] {to = "word2"} : memref<i32>
      %25 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %26 = arith.shrsi %25, %c24_i32 : i32
      %27 = arith.andi %26, %c255_i32 : i32
      %alloc_11 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %27, %alloc_11[] {to = "opcode"} : memref<i32>
      %28 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %29 = arith.shrsi %28, %c20_i32 : i32
      %30 = arith.andi %29, %c15_i32 : i32
      %alloc_12 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %30, %alloc_12[] {to = "dst"} : memref<i32>
      %31 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %32 = arith.shrsi %31, %c16_i32 : i32
      %33 = arith.andi %32, %c15_i32 : i32
      %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %33, %alloc_13[] {to = "src"} : memref<i32>
      %34 = affine.load %alloc_10[] {from = "word2"} : memref<i32>
      %35 = arith.andi %34, %c65535_i32 : i32
      %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %35, %alloc_14[] {to = "imm"} : memref<i32>
      %36 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_15 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %36, %alloc_15[] {to = "d"} : memref<i32>
      %37 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
      %38 = arith.cmpi eq, %37, %c1_i32 : i32
      scf.if %38 {
        %53 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %53, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %53 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c2_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %55, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %55 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c3_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %57, %alloc_15[] {to = "d"} : memref<i32>
          }
        }
      }
      %39 = affine.load %alloc_5[] {from = "r0"} : memref<i32>
      %alloc_16 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %39, %alloc_16[] {to = "a"} : memref<i32>
      %40 = affine.load %alloc_13[] {from = "src"} : memref<i32>
      %41 = arith.cmpi eq, %40, %c1_i32 : i32
      scf.if %41 {
        %53 = affine.load %alloc_6[] {from = "r1"} : memref<i32>
        affine.store %53, %alloc_16[] {to = "a"} : memref<i32>
      } else {
        %53 = affine.load %alloc_13[] {from = "src"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c2_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_7[] {from = "r2"} : memref<i32>
          affine.store %55, %alloc_16[] {to = "a"} : memref<i32>
        } else {
          %55 = affine.load %alloc_13[] {from = "src"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c3_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_8[] {from = "r3"} : memref<i32>
            affine.store %57, %alloc_16[] {to = "a"} : memref<i32>
          }
        }
      }
      %alloc_17 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32, %alloc_17[] {to = "wr"} : memref<i32>
      %42 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
      %43 = arith.cmpi eq, %42, %c9_i32 : i32
      scf.if %43 {
        %53 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_18 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %53, %alloc_18[] {to = "zz"} : memref<i32>
        %54 = affine.load %alloc_15[] {from = "d"} : memref<i32>
        %55 = affine.load %alloc_18[] {from = "zz"} : memref<i32>
        %56 = arith.extsi %54 : i32 to i33
        %57 = arith.extsi %55 : i32 to i33
        %58 = arith.addi %56, %57 : i33
        %59 = arith.trunci %58 : i33 to i32
        affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
      } else {
        %53 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c1_i32 : i32
        scf.if %54 {
          %55 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_18 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %55, %alloc_18[] {to = "z2"} : memref<i32>
          %56 = affine.load %alloc_18[] {from = "z2"} : memref<i32>
          affine.store %56, %alloc_15[] {to = "d"} : memref<i32>
        } else {
          %55 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c2_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_13[] {from = "src"} : memref<i32>
            %58 = arith.index_cast %57 : i32 to index
            %59 = memref.load %arg0[%arg1, %58] {from = "local_Bias"} : memref<4x2xi32, #map>
            affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
          } else {
            %57 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
            %58 = arith.cmpi eq, %57, %c3_i32 : i32
            scf.if %58 {
              %59 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              affine.store %59, %alloc_15[] {to = "d"} : memref<i32>
            } else {
              %59 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
              %60 = arith.cmpi eq, %59, %c4_i32 : i32
              scf.if %60 {
                %61 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                %62 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                %63 = arith.extsi %61 : i32 to i33
                %64 = arith.extsi %62 : i32 to i33
                %65 = arith.addi %63, %64 : i33
                %66 = arith.trunci %65 : i33 to i32
                affine.store %66, %alloc_15[] {to = "d"} : memref<i32>
              } else {
                %61 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                %62 = arith.cmpi eq, %61, %c5_i32 : i32
                scf.if %62 {
                  %63 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                  %64 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                  %65 = arith.extsi %63 : i32 to i64
                  %66 = arith.extsi %64 : i32 to i64
                  %67 = arith.muli %65, %66 : i64
                  %68 = arith.trunci %67 : i64 to i32
                  affine.store %68, %alloc_15[] {to = "d"} : memref<i32>
                } else {
                  %63 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                  %64 = arith.cmpi eq, %63, %c6_i32 : i32
                  scf.if %64 {
                    %65 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                    %66 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                    %67 = arith.cmpi sgt, %65, %66 : i32
                    scf.if %67 {
                      %68 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                      affine.store %68, %alloc_15[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %65 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                    %66 = arith.cmpi eq, %65, %c7_i32 : i32
                    scf.if %66 {
                      %67 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                      %68 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %69 = arith.shrsi %67, %68 : i32
                      affine.store %69, %alloc_15[] {to = "d"} : memref<i32>
                    } else {
                      %67 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                      %68 = arith.cmpi eq, %67, %c10_i32 : i32
                      scf.if %68 {
                        %69 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                        %70 = affine.load %alloc_16[] {from = "a"} : memref<i32>
                        %71 = arith.extsi %69 : i32 to i33
                        %72 = arith.extsi %70 : i32 to i33
                        %73 = arith.subi %71, %72 : i33
                        %74 = arith.trunci %73 : i33 to i32
                        affine.store %74, %alloc_15[] {to = "d"} : memref<i32>
                      } else {
                        %69 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                        %70 = arith.cmpi eq, %69, %c11_i32 : i32
                        scf.if %70 {
                          %71 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                          %alloc_18 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %71, %alloc_18[] {to = "e"} : memref<i32>
                          %72 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %73 = arith.cmpi slt, %72, %c0_i32 : i32
                          scf.if %73 {
                            affine.store %c0_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %74 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %75 = arith.cmpi sgt, %74, %c30_i32 : i32
                          scf.if %75 {
                            affine.store %c30_i32, %alloc_18[] {to = "e"} : memref<i32>
                          }
                          %76 = affine.load %alloc_18[] {from = "e"} : memref<i32>
                          %77 = arith.shli %c1_i32, %76 : i32
                          affine.store %77, %alloc_15[] {to = "d"} : memref<i32>
                        } else {
                          %71 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                          %72 = arith.cmpi eq, %71, %c12_i32 : i32
                          scf.if %72 {
                            %73 = affine.load %alloc_4[] {from = "rcp"} : memref<i32>
                            affine.store %73, %alloc_15[] {to = "d"} : memref<i32>
                          } else {
                            %73 = affine.load %alloc_11[] {from = "opcode"} : memref<i32>
                            %74 = arith.cmpi eq, %73, %c8_i32 : i32
                            scf.if %74 {
                              %75 = affine.load %alloc_15[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg4, [], %75) : !allo.stream<i32, 4> contains i32
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
                            } else {
                              affine.store %c0_i32, %alloc_17[] {to = "wr"} : memref<i32>
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
      %44 = affine.load %alloc_17[] {from = "wr"} : memref<i32>
      %45 = arith.cmpi eq, %44, %c1_i32 : i32
      scf.if %45 {
        %53 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
        %54 = arith.cmpi eq, %53, %c0_i32 : i32
        scf.if %54 {
          %55 = affine.load %alloc_15[] {from = "d"} : memref<i32>
          affine.store %55, %alloc_5[] {to = "r0"} : memref<i32>
        } else {
          %55 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
          %56 = arith.cmpi eq, %55, %c1_i32 : i32
          scf.if %56 {
            %57 = affine.load %alloc_15[] {from = "d"} : memref<i32>
            affine.store %57, %alloc_6[] {to = "r1"} : memref<i32>
          } else {
            %57 = affine.load %alloc_12[] {from = "dst"} : memref<i32>
            %58 = arith.cmpi eq, %57, %c2_i32 : i32
            scf.if %58 {
              %59 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %59, %alloc_7[] {to = "r2"} : memref<i32>
            } else {
              %59 = affine.load %alloc_15[] {from = "d"} : memref<i32>
              affine.store %59, %alloc_8[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %46 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %47 = arith.extsi %46 : i32 to i33
      %48 = arith.addi %47, %c1_i33 : i33
      %49 = arith.trunci %48 : i33 to i32
      affine.store %49, %alloc_9[] {to = "pc2"} : memref<i32>
      %50 = affine.load %alloc_9[] {from = "pc2"} : memref<i32>
      %51 = affine.load %alloc_0[] {from = "plen"} : memref<i32>
      %52 = arith.cmpi eq, %50, %51 : i32
      scf.if %52 {
        affine.store %c0_i32, %alloc_9[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_y_out_drain(%arg0: memref<4x4xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 4> -> i32
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<4x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<4x4xi8, #map>, %arg1: memref<5x4xi32, #map>, %arg2: memref<17xi32, #map1>, %arg3: memref<4x4x4xi8, #map2>, %arg4: memref<4x2xi32, #map>, %arg5: memref<4x4xi32, #map>) attributes {dataflow, itypes = "ssssss", otypes = "", top} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %0 = allo.stream_construct() {name = "vpu_y_out_bind_3"} : !allo.stream<i32, 4>
    %1 = allo.stream_construct() {name = "vpu_y_out_bind_2"} : !allo.stream<i32, 4>
    %2 = allo.stream_construct() {name = "vpu_op_out_op_in_3"} : !allo.stream<i32, 2>
    %3 = allo.stream_construct() {name = "vpu_y_out_bind_1"} : !allo.stream<i32, 4>
    %4 = allo.stream_construct() {name = "vpu_op_out_op_in_2"} : !allo.stream<i32, 2>
    %5 = allo.stream_construct() {name = "vpu_y_out_bind_0"} : !allo.stream<i32, 4>
    %6 = allo.stream_construct() {name = "vpu_op_out_op_in_1"} : !allo.stream<i32, 2>
    %7 = allo.stream_construct() {name = "vpu_z_in_bind_3"} : !allo.stream<i32, 2>
    %8 = allo.stream_construct() {name = "vpu_z_in_bind_2"} : !allo.stream<i32, 2>
    %9 = allo.stream_construct() {name = "mac_op_out_op_in_3_3"} : !allo.stream<i32, 2>
    %10 = allo.stream_construct() {name = "mac_a_out_a_in_3_3"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "vpu_z_in_bind_1"} : !allo.stream<i32, 2>
    %12 = allo.stream_construct() {name = "mac_op_out_op_in_3_2"} : !allo.stream<i32, 2>
    %13 = allo.stream_construct() {name = "mac_a_out_a_in_3_2"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "vpu_z_in_bind_0"} : !allo.stream<i32, 2>
    %15 = allo.stream_construct() {name = "mac_op_out_op_in_3_1"} : !allo.stream<i32, 2>
    %16 = allo.stream_construct() {name = "mac_a_out_a_in_3_1"} : !allo.stream<i8, 2>
    %17 = allo.stream_construct() {name = "mac_p_out_p_in_3_3"} : !allo.stream<i32, 2>
    %18 = allo.stream_construct() {name = "mac_p_out_p_in_3_2"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "mac_op_out_op_in_2_3"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "mac_a_out_a_in_2_3"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "mac_p_out_p_in_3_1"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "mac_op_out_op_in_2_2"} : !allo.stream<i32, 2>
    %23 = allo.stream_construct() {name = "mac_a_out_a_in_2_2"} : !allo.stream<i8, 2>
    %24 = allo.stream_construct() {name = "mac_p_out_p_in_3_0"} : !allo.stream<i32, 2>
    %25 = allo.stream_construct() {name = "mac_op_out_op_in_2_1"} : !allo.stream<i32, 2>
    %26 = allo.stream_construct() {name = "mac_a_out_a_in_2_1"} : !allo.stream<i8, 2>
    %27 = allo.stream_construct() {name = "mac_p_out_p_in_2_3"} : !allo.stream<i32, 2>
    %28 = allo.stream_construct() {name = "mac_p_out_p_in_2_2"} : !allo.stream<i32, 2>
    %29 = allo.stream_construct() {name = "mac_op_out_op_in_1_3"} : !allo.stream<i32, 2>
    %30 = allo.stream_construct() {name = "mac_a_out_a_in_1_3"} : !allo.stream<i8, 2>
    %31 = allo.stream_construct() {name = "mac_p_out_p_in_2_1"} : !allo.stream<i32, 2>
    %32 = allo.stream_construct() {name = "mac_op_out_op_in_1_2"} : !allo.stream<i32, 2>
    %33 = allo.stream_construct() {name = "mac_a_out_a_in_1_2"} : !allo.stream<i8, 2>
    %34 = allo.stream_construct() {name = "mac_p_out_p_in_2_0"} : !allo.stream<i32, 2>
    %35 = allo.stream_construct() {name = "mac_op_out_op_in_1_1"} : !allo.stream<i32, 2>
    %36 = allo.stream_construct() {name = "mac_a_out_a_in_1_1"} : !allo.stream<i8, 2>
    %37 = allo.stream_construct() {name = "mac_p_out_p_in_1_3"} : !allo.stream<i32, 2>
    %38 = allo.stream_construct() {name = "mac_p_out_p_in_1_2"} : !allo.stream<i32, 2>
    %39 = allo.stream_construct() {name = "mac_op_out_op_in_0_3"} : !allo.stream<i32, 2>
    %40 = allo.stream_construct() {name = "mac_a_out_a_in_0_3"} : !allo.stream<i8, 2>
    %41 = allo.stream_construct() {name = "mac_p_out_p_in_1_1"} : !allo.stream<i32, 2>
    %42 = allo.stream_construct() {name = "mac_op_out_op_in_0_2"} : !allo.stream<i32, 2>
    %43 = allo.stream_construct() {name = "mac_a_out_a_in_0_2"} : !allo.stream<i8, 2>
    %44 = allo.stream_construct() {name = "mac_p_out_p_in_1_0"} : !allo.stream<i32, 2>
    %45 = allo.stream_construct() {name = "mac_op_out_op_in_0_1"} : !allo.stream<i32, 2>
    %46 = allo.stream_construct() {name = "mac_a_out_a_in_0_1"} : !allo.stream<i8, 2>
    %47 = allo.stream_construct() {name = "vpu_op_in_bind_0"} : !allo.stream<i32, 17>
    %48 = allo.stream_construct() {name = "mac_op_in_bind_3"} : !allo.stream<i32, 5>
    %49 = allo.stream_construct() {name = "mac_op_in_bind_2"} : !allo.stream<i32, 5>
    %50 = allo.stream_construct() {name = "mac_op_in_bind_1"} : !allo.stream<i32, 5>
    %51 = allo.stream_construct() {name = "mac_op_in_bind_0"} : !allo.stream<i32, 5>
    %52 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 4>
    %53 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 4>
    %54 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 4>
    %55 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 4>
    call @mac_a_in_load(%arg0, %c0, %55) : (memref<4x4xi8, #map>, index, !allo.stream<i8, 4>) -> ()
    call @mac_a_in_load(%arg0, %c1, %54) : (memref<4x4xi8, #map>, index, !allo.stream<i8, 4>) -> ()
    call @mac_a_in_load(%arg0, %c2, %53) : (memref<4x4xi8, #map>, index, !allo.stream<i8, 4>) -> ()
    call @mac_a_in_load(%arg0, %c3, %52) : (memref<4x4xi8, #map>, index, !allo.stream<i8, 4>) -> ()
    call @mac_op_in_load(%arg1, %c0, %51) : (memref<5x4xi32, #map>, index, !allo.stream<i32, 5>) -> ()
    call @mac_op_in_load(%arg1, %c1, %50) : (memref<5x4xi32, #map>, index, !allo.stream<i32, 5>) -> ()
    call @mac_op_in_load(%arg1, %c2, %49) : (memref<5x4xi32, #map>, index, !allo.stream<i32, 5>) -> ()
    call @mac_op_in_load(%arg1, %c3, %48) : (memref<5x4xi32, #map>, index, !allo.stream<i32, 5>) -> ()
    call @vpu_op_in_load(%arg2, %c0, %47) : (memref<17xi32, #map1>, index, !allo.stream<i32, 17>) -> ()
    call @mac_r8(%arg3, %c0, %c0, %55, %46, %51, %45, %44) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 4>, !allo.stream<i8, 2>, !allo.stream<i32, 5>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg3, %c0, %c1, %46, %43, %45, %42, %41) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg3, %c0, %c2, %43, %40, %42, %39, %38) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r6(%arg3, %c0, %c3, %40, %39, %37) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg3, %c1, %c0, %54, %36, %50, %35, %44, %34) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 4>, !allo.stream<i8, 2>, !allo.stream<i32, 5>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg3, %c1, %c1, %36, %33, %35, %32, %41, %31) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg3, %c1, %c2, %33, %30, %32, %29, %38, %28) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg3, %c1, %c3, %30, %29, %37, %27) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg3, %c2, %c0, %53, %26, %49, %25, %34, %24) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 4>, !allo.stream<i8, 2>, !allo.stream<i32, 5>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg3, %c2, %c1, %26, %23, %25, %22, %31, %21) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg3, %c2, %c2, %23, %20, %22, %19, %28, %18) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg3, %c2, %c3, %20, %19, %27, %17) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r7(%arg3, %c3, %c0, %52, %16, %48, %15, %24, %14) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 4>, !allo.stream<i8, 2>, !allo.stream<i32, 5>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg3, %c3, %c1, %16, %13, %15, %12, %21, %11) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg3, %c3, %c2, %13, %10, %12, %9, %18, %8) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r5(%arg3, %c3, %c3, %10, %9, %17, %7) : (memref<4x4x4xi8, #map2>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_r2(%arg4, %c0, %47, %6, %5, %14) : (memref<4x2xi32, #map>, index, !allo.stream<i32, 17>, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i32, 2>) -> ()
    call @vpu_r0(%arg4, %c1, %6, %4, %3, %11) : (memref<4x2xi32, #map>, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i32, 2>) -> ()
    call @vpu_r0(%arg4, %c2, %4, %2, %1, %8) : (memref<4x2xi32, #map>, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i32, 2>) -> ()
    call @vpu_r1(%arg4, %c3, %2, %0, %7) : (memref<4x2xi32, #map>, index, !allo.stream<i32, 2>, !allo.stream<i32, 4>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain(%arg5, %c0, %5) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @vpu_y_out_drain(%arg5, %c1, %3) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @vpu_y_out_drain(%arg5, %c2, %1) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    call @vpu_y_out_drain(%arg5, %c3, %0) : (memref<4x4xi32, #map>, index, !allo.stream<i32, 4>) -> ()
    return
  }
}
