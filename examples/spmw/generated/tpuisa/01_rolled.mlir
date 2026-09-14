module {
  func.func @mac_a_in_load(%arg0: memref<4x4xi8>, %arg1: index, %arg2: !allo.stream<i8, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<4x4xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 4> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @mac_op_in_load(%arg0: memref<5x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 5>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 5 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_MProg"} : memref<5x4xi32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 5> contains i32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @vpu_op_in_load(%arg0: memref<17xi32>, %arg1: index, %arg2: !allo.stream<i32, 17>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 17 {
      %0 = affine.load %arg0[%arg3] {from = "local_VProg"} : memref<17xi32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 17> contains i32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @mac_r0(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg9 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %15 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %15, %alloc_11[] {to = "p"} : memref<i32>
      %16 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %16) : !allo.stream<i8, 2> contains i8
      %17 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg0[%arg1, %arg2, %18] {from = "local_W"} : memref<4x4x4xi8>
      %20 = arith.extsi %19 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %20, %alloc_12[] {to = "wt"} : memref<i32>
      %21 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %22 = arith.cmpi eq, %21, %c1_i32_14 : i32
      scf.if %22 {
        %23 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %25 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %26 = arith.extsi %24 : i8 to i40
        %27 = arith.extsi %25 : i32 to i40
        %28 = arith.muli %26, %27 : i40
        %29 = arith.extsi %23 : i32 to i41
        %30 = arith.extsi %28 : i40 to i41
        %31 = arith.addi %29, %30 : i41
        allo.stream_put(%arg8, [], %31) : !allo.stream<i32, 2> contains i41
      } else {
        %23 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %24 = arith.cmpi eq, %23, %c2_i32_15 : i32
        scf.if %24 {
          %25 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %26 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %27 = arith.extsi %25 : i8 to i40
          %28 = arith.extsi %26 : i32 to i40
          %29 = arith.muli %27, %28 : i40
          allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i40
        } else {
          %25 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %25) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r1(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg9 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %15 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %15, %alloc_11[] {to = "p"} : memref<i32>
      %16 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %16) : !allo.stream<i8, 2> contains i8
      %17 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg0[%arg1, %arg2, %18] {from = "local_W"} : memref<4x4x4xi8>
      %20 = arith.extsi %19 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %20, %alloc_12[] {to = "wt"} : memref<i32>
      %21 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %22 = arith.cmpi eq, %21, %c1_i32_14 : i32
      scf.if %22 {
        %23 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %25 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %26 = arith.extsi %24 : i8 to i40
        %27 = arith.extsi %25 : i32 to i40
        %28 = arith.muli %26, %27 : i40
        %29 = arith.extsi %23 : i32 to i41
        %30 = arith.extsi %28 : i40 to i41
        %31 = arith.addi %29, %30 : i41
        allo.stream_put(%arg8, [], %31) : !allo.stream<i32, 2> contains i41
      } else {
        %23 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %24 = arith.cmpi eq, %23, %c2_i32_15 : i32
        scf.if %24 {
          %25 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %26 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %27 = arith.extsi %25 : i8 to i40
          %28 = arith.extsi %26 : i32 to i40
          %29 = arith.muli %27, %28 : i40
          allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i40
        } else {
          %25 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %25) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r2(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_______", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg8 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %c0_i32_11 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      %alloc_13 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_12, %alloc_13[] {to = "p"} : memref<i32>
      %15 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %15) : !allo.stream<i8, 2> contains i8
      %16 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %arg0[%arg1, %arg2, %17] {from = "local_W"} : memref<4x4x4xi8>
      %19 = arith.extsi %18 : i8 to i32
      %alloc_14 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %19, %alloc_14[] {to = "wt"} : memref<i32>
      %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_15 = arith.constant 1 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %21 = arith.cmpi eq, %20, %c1_i32_16 : i32
      scf.if %21 {
        %22 = affine.load %alloc_13[] {from = "p"} : memref<i32>
        %23 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %24 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
        %25 = arith.extsi %23 : i8 to i40
        %26 = arith.extsi %24 : i32 to i40
        %27 = arith.muli %25, %26 : i40
        %28 = arith.extsi %22 : i32 to i41
        %29 = arith.extsi %27 : i40 to i41
        %30 = arith.addi %28, %29 : i41
        allo.stream_put(%arg7, [], %30) : !allo.stream<i32, 2> contains i41
      } else {
        %22 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_17 = arith.constant 2 : i32
        %23 = arith.cmpi eq, %22, %c2_i32_17 : i32
        scf.if %23 {
          %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %25 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
          %26 = arith.extsi %24 : i8 to i40
          %27 = arith.extsi %25 : i32 to i40
          %28 = arith.muli %26, %27 : i40
          allo.stream_put(%arg7, [], %28) : !allo.stream<i32, 2> contains i40
        } else {
          %24 = affine.load %alloc_13[] {from = "p"} : memref<i32>
          allo.stream_put(%arg7, [], %24) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r3(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %0 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.index_cast %c0_i32_0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %4 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg7 = %2 to %3 step %4 {
      %5 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %5, %alloc_2[] {to = "word"} : memref<i32>
      %6 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %7 = arith.shrsi %6, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %8 = arith.andi %7, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_5[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %10 = arith.shrsi %9, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %11 = arith.andi %10, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_9[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_10[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_11[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %arg0[%arg1, %arg2, %15] {from = "local_W"} : memref<4x4x4xi8>
      %17 = arith.extsi %16 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %17, %alloc_12[] {to = "wt"} : memref<i32>
      %18 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %19 = arith.cmpi eq, %18, %c1_i32_14 : i32
      scf.if %19 {
        %20 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %21 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %22 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %23 = arith.extsi %21 : i8 to i40
        %24 = arith.extsi %22 : i32 to i40
        %25 = arith.muli %23, %24 : i40
        %26 = arith.extsi %20 : i32 to i41
        %27 = arith.extsi %25 : i40 to i41
        %28 = arith.addi %26, %27 : i41
        allo.stream_put(%arg6, [], %28) : !allo.stream<i32, 2> contains i41
      } else {
        %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %21 = arith.cmpi eq, %20, %c2_i32_15 : i32
        scf.if %21 {
          %22 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %23 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %24 = arith.extsi %22 : i8 to i40
          %25 = arith.extsi %23 : i32 to i40
          %26 = arith.muli %24, %25 : i40
          allo.stream_put(%arg6, [], %26) : !allo.stream<i32, 2> contains i40
        } else {
          %22 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %22) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r4(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg9 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %15 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %15, %alloc_11[] {to = "p"} : memref<i32>
      %16 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %16) : !allo.stream<i8, 2> contains i8
      %17 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg0[%arg1, %arg2, %18] {from = "local_W"} : memref<4x4x4xi8>
      %20 = arith.extsi %19 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %20, %alloc_12[] {to = "wt"} : memref<i32>
      %21 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %22 = arith.cmpi eq, %21, %c1_i32_14 : i32
      scf.if %22 {
        %23 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %25 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %26 = arith.extsi %24 : i8 to i40
        %27 = arith.extsi %25 : i32 to i40
        %28 = arith.muli %26, %27 : i40
        %29 = arith.extsi %23 : i32 to i41
        %30 = arith.extsi %28 : i40 to i41
        %31 = arith.addi %29, %30 : i41
        allo.stream_put(%arg8, [], %31) : !allo.stream<i32, 2> contains i41
      } else {
        %23 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %24 = arith.cmpi eq, %23, %c2_i32_15 : i32
        scf.if %24 {
          %25 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %26 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %27 = arith.extsi %25 : i8 to i40
          %28 = arith.extsi %26 : i32 to i40
          %29 = arith.muli %27, %28 : i40
          allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i40
        } else {
          %25 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %25) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r5(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    %0 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.index_cast %c0_i32_0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %4 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg7 = %2 to %3 step %4 {
      %5 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %5, %alloc_2[] {to = "word"} : memref<i32>
      %6 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %7 = arith.shrsi %6, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %8 = arith.andi %7, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_5[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %10 = arith.shrsi %9, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %11 = arith.andi %10, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_9[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_10[] {to = "a"} : memref<i8>
      %13 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %13, %alloc_11[] {to = "p"} : memref<i32>
      %14 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %arg0[%arg1, %arg2, %15] {from = "local_W"} : memref<4x4x4xi8>
      %17 = arith.extsi %16 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %17, %alloc_12[] {to = "wt"} : memref<i32>
      %18 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %19 = arith.cmpi eq, %18, %c1_i32_14 : i32
      scf.if %19 {
        %20 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %21 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %22 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %23 = arith.extsi %21 : i8 to i40
        %24 = arith.extsi %22 : i32 to i40
        %25 = arith.muli %23, %24 : i40
        %26 = arith.extsi %20 : i32 to i41
        %27 = arith.extsi %25 : i40 to i41
        %28 = arith.addi %26, %27 : i41
        allo.stream_put(%arg6, [], %28) : !allo.stream<i32, 2> contains i41
      } else {
        %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %21 = arith.cmpi eq, %20, %c2_i32_15 : i32
        scf.if %21 {
          %22 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %23 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %24 = arith.extsi %22 : i8 to i40
          %25 = arith.extsi %23 : i32 to i40
          %26 = arith.muli %24, %25 : i40
          allo.stream_put(%arg6, [], %26) : !allo.stream<i32, 2> contains i40
        } else {
          %22 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg6, [], %22) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r6(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %0 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %2 = arith.index_cast %c0_i32_0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %4 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg6 = %2 to %3 step %4 {
      %5 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %5, %alloc_2[] {to = "word"} : memref<i32>
      %6 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %7 = arith.shrsi %6, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %8 = arith.andi %7, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %8, %alloc_5[] {to = "opcode"} : memref<i32>
      %9 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %10 = arith.shrsi %9, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %11 = arith.andi %10, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %11, %alloc_9[] {to = "tile"} : memref<i32>
      %12 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %12, %alloc_10[] {to = "a"} : memref<i8>
      %c0_i32_11 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      %alloc_13 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_12, %alloc_13[] {to = "p"} : memref<i32>
      %13 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %14 = arith.index_cast %13 : i32 to index
      %15 = memref.load %arg0[%arg1, %arg2, %14] {from = "local_W"} : memref<4x4x4xi8>
      %16 = arith.extsi %15 : i8 to i32
      %alloc_14 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %16, %alloc_14[] {to = "wt"} : memref<i32>
      %17 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_15 = arith.constant 1 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %18 = arith.cmpi eq, %17, %c1_i32_16 : i32
      scf.if %18 {
        %19 = affine.load %alloc_13[] {from = "p"} : memref<i32>
        %20 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %21 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
        %22 = arith.extsi %20 : i8 to i40
        %23 = arith.extsi %21 : i32 to i40
        %24 = arith.muli %22, %23 : i40
        %25 = arith.extsi %19 : i32 to i41
        %26 = arith.extsi %24 : i40 to i41
        %27 = arith.addi %25, %26 : i41
        allo.stream_put(%arg5, [], %27) : !allo.stream<i32, 2> contains i41
      } else {
        %19 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_17 = arith.constant 2 : i32
        %20 = arith.cmpi eq, %19, %c2_i32_17 : i32
        scf.if %20 {
          %21 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %22 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
          %23 = arith.extsi %21 : i8 to i40
          %24 = arith.extsi %22 : i32 to i40
          %25 = arith.muli %23, %24 : i40
          allo.stream_put(%arg5, [], %25) : !allo.stream<i32, 2> contains i40
        } else {
          %21 = affine.load %alloc_13[] {from = "p"} : memref<i32>
          allo.stream_put(%arg5, [], %21) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r7(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>, %arg8: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s________", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg9 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %15 = allo.stream_get(%arg7, []) : !allo.stream<i32, 2> -> i32
      %alloc_11 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %15, %alloc_11[] {to = "p"} : memref<i32>
      %16 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %16) : !allo.stream<i8, 2> contains i8
      %17 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg0[%arg1, %arg2, %18] {from = "local_W"} : memref<4x4x4xi8>
      %20 = arith.extsi %19 : i8 to i32
      %alloc_12 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %20, %alloc_12[] {to = "wt"} : memref<i32>
      %21 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %c1_i32_14 = arith.constant 1 : i32
      %22 = arith.cmpi eq, %21, %c1_i32_14 : i32
      scf.if %22 {
        %23 = affine.load %alloc_11[] {from = "p"} : memref<i32>
        %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %25 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
        %26 = arith.extsi %24 : i8 to i40
        %27 = arith.extsi %25 : i32 to i40
        %28 = arith.muli %26, %27 : i40
        %29 = arith.extsi %23 : i32 to i41
        %30 = arith.extsi %28 : i40 to i41
        %31 = arith.addi %29, %30 : i41
        allo.stream_put(%arg8, [], %31) : !allo.stream<i32, 2> contains i41
      } else {
        %23 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_15 = arith.constant 2 : i32
        %24 = arith.cmpi eq, %23, %c2_i32_15 : i32
        scf.if %24 {
          %25 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %26 = affine.load %alloc_12[] {from = "wt"} : memref<i32>
          %27 = arith.extsi %25 : i8 to i40
          %28 = arith.extsi %26 : i32 to i40
          %29 = arith.muli %27, %28 : i40
          allo.stream_put(%arg8, [], %29) : !allo.stream<i32, 2> contains i40
        } else {
          %25 = affine.load %alloc_11[] {from = "p"} : memref<i32>
          allo.stream_put(%arg8, [], %25) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @mac_r8(%arg0: memref<4x4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 4>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 5>, %arg6: !allo.stream<i32, 2>, %arg7: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_______", otypes = ""} {
    %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
    %alloc = memref.alloc() {name = "count"} : memref<i32>
    affine.store %0, %alloc[] {to = "count"} : memref<i32>
    %1 = affine.load %alloc[] {from = "count"} : memref<i32>
    allo.stream_put(%arg6, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "count"} : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %3 = arith.index_cast %c0_i32_0 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.index_cast %c1_i32_1 : i32 to index
    scf.for %arg8 = %3 to %4 step %5 {
      %6 = allo.stream_get(%arg5, []) : !allo.stream<i32, 5> -> i32
      %alloc_2 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %6, %alloc_2[] {to = "word"} : memref<i32>
      %7 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      allo.stream_put(%arg6, [], %7) : !allo.stream<i32, 2> contains i32
      %8 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_3 = arith.constant 24 : i32
      %9 = arith.shrsi %8, %c24_i32_3 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_4 = arith.constant 255 : i32
      %10 = arith.andi %9, %c255_i32_4 : i32
      %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %10, %alloc_5[] {to = "opcode"} : memref<i32>
      %11 = affine.load %alloc_2[] {from = "word"} : memref<i32>
      %c16_i32 = arith.constant 16 : i32
      %c16_i32_6 = arith.constant 16 : i32
      %12 = arith.shrsi %11, %c16_i32_6 : i32
      %c255_i32_7 = arith.constant 255 : i32
      %c255_i32_8 = arith.constant 255 : i32
      %13 = arith.andi %12, %c255_i32_8 : i32
      %alloc_9 = memref.alloc() {name = "tile"} : memref<i32>
      affine.store %13, %alloc_9[] {to = "tile"} : memref<i32>
      %14 = allo.stream_get(%arg3, []) : !allo.stream<i8, 4> -> i8
      %alloc_10 = memref.alloc() {name = "a"} : memref<i8>
      affine.store %14, %alloc_10[] {to = "a"} : memref<i8>
      %c0_i32_11 = arith.constant 0 : i32
      %c0_i32_12 = arith.constant 0 : i32
      %alloc_13 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_12, %alloc_13[] {to = "p"} : memref<i32>
      %15 = affine.load %alloc_10[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %15) : !allo.stream<i8, 2> contains i8
      %16 = affine.load %alloc_9[] {from = "tile"} : memref<i32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %arg0[%arg1, %arg2, %17] {from = "local_W"} : memref<4x4x4xi8>
      %19 = arith.extsi %18 : i8 to i32
      %alloc_14 = memref.alloc() {name = "wt"} : memref<i32>
      affine.store %19, %alloc_14[] {to = "wt"} : memref<i32>
      %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
      %c1_i32_15 = arith.constant 1 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %21 = arith.cmpi eq, %20, %c1_i32_16 : i32
      scf.if %21 {
        %22 = affine.load %alloc_13[] {from = "p"} : memref<i32>
        %23 = affine.load %alloc_10[] {from = "a"} : memref<i8>
        %24 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
        %25 = arith.extsi %23 : i8 to i40
        %26 = arith.extsi %24 : i32 to i40
        %27 = arith.muli %25, %26 : i40
        %28 = arith.extsi %22 : i32 to i41
        %29 = arith.extsi %27 : i40 to i41
        %30 = arith.addi %28, %29 : i41
        allo.stream_put(%arg7, [], %30) : !allo.stream<i32, 2> contains i41
      } else {
        %22 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_17 = arith.constant 2 : i32
        %23 = arith.cmpi eq, %22, %c2_i32_17 : i32
        scf.if %23 {
          %24 = affine.load %alloc_10[] {from = "a"} : memref<i8>
          %25 = affine.load %alloc_14[] {from = "wt"} : memref<i32>
          %26 = arith.extsi %24 : i8 to i40
          %27 = arith.extsi %25 : i32 to i40
          %28 = arith.muli %26, %27 : i40
          allo.stream_put(%arg7, [], %28) : !allo.stream<i32, 2> contains i40
        } else {
          %24 = affine.load %alloc_13[] {from = "p"} : memref<i32>
          allo.stream_put(%arg7, [], %24) : !allo.stream<i32, 2> contains i32
        }
      }
    } {loop_name = "step", op_name = "S_step_0"}
    return
  }
  func.func @vpu_r0(%arg0: memref<4x2xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 4>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c65535_i32 = arith.constant 65535 : i32
    %c65535_i32_0 = arith.constant 65535 : i32
    %3 = arith.andi %2, %c65535_i32_0 : i32
    %alloc_1 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %3, %alloc_1[] {to = "plen"} : memref<i32>
    %4 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_2 = arith.constant 16 : i32
    %5 = arith.shrsi %4, %c16_i32_2 : i32
    %c65535_i32_3 = arith.constant 65535 : i32
    %c65535_i32_4 = arith.constant 65535 : i32
    %6 = arith.andi %5, %c65535_i32_4 : i32
    %alloc_5 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %6, %alloc_5[] {to = "nouts"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "prog"} : memref<16xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<16xi32>)
    %7 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %8 = arith.index_cast %c0_i32_8 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %10 = arith.index_cast %c1_i32_9 : i32 to index
    scf.for %arg6 = %8 to %9 step %10 {
      %29 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_41 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %29, %alloc_41[] {to = "word"} : memref<i32>
      %30 = affine.load %alloc_41[] {from = "word"} : memref<i32>
      memref.store %30, %alloc_6[%arg6] {to = "prog"} : memref<16xi32>
      %31 = affine.load %alloc_41[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %31) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    %11 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c16_i32_10 = arith.constant 16 : i32
    %c16_i32_11 = arith.constant 16 : i32
    %12 = arith.extsi %c16_i32_11 : i32 to i33
    %13 = arith.extsi %11 : i32 to i33
    %14 = arith.subi %12, %13 : i33
    %c0_i32_12 = arith.constant 0 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %15 = arith.index_cast %c0_i32_13 : i32 to index
    %16 = arith.index_cast %14 : i33 to index
    %c1_i32_14 = arith.constant 1 : i32
    %c1_i32_15 = arith.constant 1 : i32
    %17 = arith.index_cast %c1_i32_15 : i32 to index
    scf.for %arg6 = %15 to %16 step %17 {
      %29 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_41 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %29, %alloc_41[] {to = "spare"} : memref<i32>
      %30 = affine.load %alloc_41[] {from = "spare"} : memref<i32>
      allo.stream_put(%arg3, [], %30) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_pad", op_name = "S__pad_1"}
    %18 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32>
    %alloc_16 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %18, %alloc_16[] {to = "denom"} : memref<i32>
    %c0_i32_17 = arith.constant 0 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %alloc_19 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32_18, %alloc_19[] {to = "rcp"} : memref<i32>
    %19 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
    %c0_i32_20 = arith.constant 0 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %20 = arith.cmpi sgt, %19, %c0_i32_21 : i32
    scf.if %20 {
      %c1_i32_41 = arith.constant 1 : i32
      %c1_i32_42 = arith.constant 1 : i32
      %c14_i32 = arith.constant 14 : i32
      %c14_i32_43 = arith.constant 14 : i32
      %29 = arith.shli %c1_i32_42, %c14_i32_43 : i32
      %30 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
      %31 = arith.floordivsi %29, %30 : i32
      affine.store %31, %alloc_19[] {to = "rcp"} : memref<i32>
    }
    %c0_i32_22 = arith.constant 0 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %alloc_24 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32_23, %alloc_24[] {to = "r0"} : memref<i32>
    %c0_i32_25 = arith.constant 0 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %alloc_27 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32_26, %alloc_27[] {to = "r1"} : memref<i32>
    %c0_i32_28 = arith.constant 0 : i32
    %c0_i32_29 = arith.constant 0 : i32
    %alloc_30 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32_29, %alloc_30[] {to = "r2"} : memref<i32>
    %c0_i32_31 = arith.constant 0 : i32
    %c0_i32_32 = arith.constant 0 : i32
    %alloc_33 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32_32, %alloc_33[] {to = "r3"} : memref<i32>
    %c0_i32_34 = arith.constant 0 : i32
    %c0_i32_35 = arith.constant 0 : i32
    %alloc_36 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32_35, %alloc_36[] {to = "pc2"} : memref<i32>
    %21 = affine.load %alloc_5[] {from = "nouts"} : memref<i32>
    %22 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %23 = arith.extsi %21 : i32 to i64
    %24 = arith.extsi %22 : i32 to i64
    %25 = arith.muli %23, %24 : i64
    %c0_i32_37 = arith.constant 0 : i32
    %c0_i32_38 = arith.constant 0 : i32
    %26 = arith.index_cast %c0_i32_38 : i32 to index
    %27 = arith.index_cast %25 : i64 to index
    %c1_i32_39 = arith.constant 1 : i32
    %c1_i32_40 = arith.constant 1 : i32
    %28 = arith.index_cast %c1_i32_40 : i32 to index
    scf.for %arg6 = %26 to %27 step %28 {
      %29 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %alloc_6[%30] {from = "prog"} : memref<16xi32>
      %alloc_41 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %31, %alloc_41[] {to = "word2"} : memref<i32>
      %32 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_42 = arith.constant 24 : i32
      %33 = arith.shrsi %32, %c24_i32_42 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_43 = arith.constant 255 : i32
      %34 = arith.andi %33, %c255_i32_43 : i32
      %alloc_44 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %34, %alloc_44[] {to = "opcode"} : memref<i32>
      %35 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c20_i32 = arith.constant 20 : i32
      %c20_i32_45 = arith.constant 20 : i32
      %36 = arith.shrsi %35, %c20_i32_45 : i32
      %c15_i32 = arith.constant 15 : i32
      %c15_i32_46 = arith.constant 15 : i32
      %37 = arith.andi %36, %c15_i32_46 : i32
      %alloc_47 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %37, %alloc_47[] {to = "dst"} : memref<i32>
      %38 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c16_i32_48 = arith.constant 16 : i32
      %c16_i32_49 = arith.constant 16 : i32
      %39 = arith.shrsi %38, %c16_i32_49 : i32
      %c15_i32_50 = arith.constant 15 : i32
      %c15_i32_51 = arith.constant 15 : i32
      %40 = arith.andi %39, %c15_i32_51 : i32
      %alloc_52 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %40, %alloc_52[] {to = "src"} : memref<i32>
      %41 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c65535_i32_53 = arith.constant 65535 : i32
      %c65535_i32_54 = arith.constant 65535 : i32
      %42 = arith.andi %41, %c65535_i32_54 : i32
      %alloc_55 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %42, %alloc_55[] {to = "imm"} : memref<i32>
      %43 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_56 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %43, %alloc_56[] {to = "d"} : memref<i32>
      %44 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_58 = arith.constant 1 : i32
      %45 = arith.cmpi eq, %44, %c1_i32_58 : i32
      scf.if %45 {
        %61 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %61, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %61 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %62 = arith.cmpi eq, %61, %c2_i32_70 : i32
        scf.if %62 {
          %63 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %63, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %63 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %64 = arith.cmpi eq, %63, %c3_i32_71 : i32
          scf.if %64 {
            %65 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %65, %alloc_56[] {to = "d"} : memref<i32>
          }
        }
      }
      %46 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_59 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %46, %alloc_59[] {to = "a"} : memref<i32>
      %47 = affine.load %alloc_52[] {from = "src"} : memref<i32>
      %c1_i32_60 = arith.constant 1 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %48 = arith.cmpi eq, %47, %c1_i32_61 : i32
      scf.if %48 {
        %61 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %61, %alloc_59[] {to = "a"} : memref<i32>
      } else {
        %61 = affine.load %alloc_52[] {from = "src"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %62 = arith.cmpi eq, %61, %c2_i32_70 : i32
        scf.if %62 {
          %63 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %63, %alloc_59[] {to = "a"} : memref<i32>
        } else {
          %63 = affine.load %alloc_52[] {from = "src"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %64 = arith.cmpi eq, %63, %c3_i32_71 : i32
          scf.if %64 {
            %65 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %65, %alloc_59[] {to = "a"} : memref<i32>
          }
        }
      }
      %c1_i32_62 = arith.constant 1 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %alloc_64 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32_63, %alloc_64[] {to = "wr"} : memref<i32>
      %49 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
      %c9_i32 = arith.constant 9 : i32
      %c9_i32_65 = arith.constant 9 : i32
      %50 = arith.cmpi eq, %49, %c9_i32_65 : i32
      scf.if %50 {
        %61 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_70 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %61, %alloc_70[] {to = "zz"} : memref<i32>
        %62 = affine.load %alloc_56[] {from = "d"} : memref<i32>
        %63 = affine.load %alloc_70[] {from = "zz"} : memref<i32>
        %64 = arith.extsi %62 : i32 to i33
        %65 = arith.extsi %63 : i32 to i33
        %66 = arith.addi %64, %65 : i33
        %67 = arith.trunci %66 : i33 to i32
        affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %61 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
        %c1_i32_70 = arith.constant 1 : i32
        %c1_i32_71 = arith.constant 1 : i32
        %62 = arith.cmpi eq, %61, %c1_i32_71 : i32
        scf.if %62 {
          %63 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_72 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %63, %alloc_72[] {to = "z2"} : memref<i32>
          %64 = affine.load %alloc_72[] {from = "z2"} : memref<i32>
          affine.store %64, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %63 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_72 = arith.constant 2 : i32
          %64 = arith.cmpi eq, %63, %c2_i32_72 : i32
          scf.if %64 {
            %65 = affine.load %alloc_52[] {from = "src"} : memref<i32>
            %66 = arith.index_cast %65 : i32 to index
            %67 = memref.load %arg0[%arg1, %66] {from = "local_Bias"} : memref<4x2xi32>
            affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
          } else {
            %65 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_73 = arith.constant 3 : i32
            %66 = arith.cmpi eq, %65, %c3_i32_73 : i32
            scf.if %66 {
              %67 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
              affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
            } else {
              %67 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_74 = arith.constant 4 : i32
              %68 = arith.cmpi eq, %67, %c4_i32_74 : i32
              scf.if %68 {
                %69 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                %70 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                %71 = arith.extsi %69 : i32 to i33
                %72 = arith.extsi %70 : i32 to i33
                %73 = arith.addi %71, %72 : i33
                %74 = arith.trunci %73 : i33 to i32
                affine.store %74, %alloc_56[] {to = "d"} : memref<i32>
              } else {
                %69 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_75 = arith.constant 5 : i32
                %70 = arith.cmpi eq, %69, %c5_i32_75 : i32
                scf.if %70 {
                  %71 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                  %72 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                  %73 = arith.extsi %71 : i32 to i64
                  %74 = arith.extsi %72 : i32 to i64
                  %75 = arith.muli %73, %74 : i64
                  %76 = arith.trunci %75 : i64 to i32
                  affine.store %76, %alloc_56[] {to = "d"} : memref<i32>
                } else {
                  %71 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_76 = arith.constant 6 : i32
                  %72 = arith.cmpi eq, %71, %c6_i32_76 : i32
                  scf.if %72 {
                    %73 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                    %74 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                    %75 = arith.cmpi sgt, %73, %74 : i32
                    scf.if %75 {
                      %76 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                      affine.store %76, %alloc_56[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %73 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_77 = arith.constant 7 : i32
                    %74 = arith.cmpi eq, %73, %c7_i32_77 : i32
                    scf.if %74 {
                      %75 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                      %76 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
                      %77 = arith.shrsi %75, %76 : i32
                      affine.store %77, %alloc_56[] {to = "d"} : memref<i32>
                    } else {
                      %75 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                      %c10_i32 = arith.constant 10 : i32
                      %c10_i32_78 = arith.constant 10 : i32
                      %76 = arith.cmpi eq, %75, %c10_i32_78 : i32
                      scf.if %76 {
                        %77 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                        %78 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                        %79 = arith.extsi %77 : i32 to i33
                        %80 = arith.extsi %78 : i32 to i33
                        %81 = arith.subi %79, %80 : i33
                        %82 = arith.trunci %81 : i33 to i32
                        affine.store %82, %alloc_56[] {to = "d"} : memref<i32>
                      } else {
                        %77 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                        %c11_i32 = arith.constant 11 : i32
                        %c11_i32_79 = arith.constant 11 : i32
                        %78 = arith.cmpi eq, %77, %c11_i32_79 : i32
                        scf.if %78 {
                          %79 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                          %alloc_80 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %79, %alloc_80[] {to = "e"} : memref<i32>
                          %80 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c0_i32_81 = arith.constant 0 : i32
                          %c0_i32_82 = arith.constant 0 : i32
                          %81 = arith.cmpi slt, %80, %c0_i32_82 : i32
                          scf.if %81 {
                            %c0_i32_86 = arith.constant 0 : i32
                            %c0_i32_87 = arith.constant 0 : i32
                            affine.store %c0_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %82 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c30_i32 = arith.constant 30 : i32
                          %c30_i32_83 = arith.constant 30 : i32
                          %83 = arith.cmpi sgt, %82, %c30_i32_83 : i32
                          scf.if %83 {
                            %c30_i32_86 = arith.constant 30 : i32
                            %c30_i32_87 = arith.constant 30 : i32
                            affine.store %c30_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %84 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c1_i32_84 = arith.constant 1 : i32
                          %c1_i32_85 = arith.constant 1 : i32
                          %85 = arith.shli %c1_i32_85, %84 : i32
                          affine.store %85, %alloc_56[] {to = "d"} : memref<i32>
                        } else {
                          %79 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                          %c12_i32 = arith.constant 12 : i32
                          %c12_i32_80 = arith.constant 12 : i32
                          %80 = arith.cmpi eq, %79, %c12_i32_80 : i32
                          scf.if %80 {
                            %81 = affine.load %alloc_19[] {from = "rcp"} : memref<i32>
                            affine.store %81, %alloc_56[] {to = "d"} : memref<i32>
                          } else {
                            %81 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                            %c8_i32 = arith.constant 8 : i32
                            %c8_i32_81 = arith.constant 8 : i32
                            %82 = arith.cmpi eq, %81, %c8_i32_81 : i32
                            scf.if %82 {
                              %83 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg4, [], %83) : !allo.stream<i32, 4> contains i32
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
                            } else {
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
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
      %51 = affine.load %alloc_64[] {from = "wr"} : memref<i32>
      %c1_i32_66 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      %52 = arith.cmpi eq, %51, %c1_i32_67 : i32
      scf.if %52 {
        %61 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        %62 = arith.cmpi eq, %61, %c0_i32_71 : i32
        scf.if %62 {
          %63 = affine.load %alloc_56[] {from = "d"} : memref<i32>
          affine.store %63, %alloc_24[] {to = "r0"} : memref<i32>
        } else {
          %63 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c1_i32_72 = arith.constant 1 : i32
          %c1_i32_73 = arith.constant 1 : i32
          %64 = arith.cmpi eq, %63, %c1_i32_73 : i32
          scf.if %64 {
            %65 = affine.load %alloc_56[] {from = "d"} : memref<i32>
            affine.store %65, %alloc_27[] {to = "r1"} : memref<i32>
          } else {
            %65 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_74 = arith.constant 2 : i32
            %66 = arith.cmpi eq, %65, %c2_i32_74 : i32
            scf.if %66 {
              %67 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %67, %alloc_30[] {to = "r2"} : memref<i32>
            } else {
              %67 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %67, %alloc_33[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %53 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %54 = arith.extsi %53 : i32 to i33
      %c1_i32_68 = arith.constant 1 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %55 = arith.extsi %c1_i32_69 : i32 to i33
      %56 = arith.addi %54, %55 : i33
      %57 = arith.trunci %56 : i33 to i32
      affine.store %57, %alloc_36[] {to = "pc2"} : memref<i32>
      %58 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %59 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
      %60 = arith.cmpi eq, %58, %59 : i32
      scf.if %60 {
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        affine.store %c0_i32_71, %alloc_36[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2"}
    return
  }
  func.func @vpu_r1(%arg0: memref<4x2xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 4>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c65535_i32 = arith.constant 65535 : i32
    %c65535_i32_0 = arith.constant 65535 : i32
    %2 = arith.andi %1, %c65535_i32_0 : i32
    %alloc_1 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %2, %alloc_1[] {to = "plen"} : memref<i32>
    %3 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_2 = arith.constant 16 : i32
    %4 = arith.shrsi %3, %c16_i32_2 : i32
    %c65535_i32_3 = arith.constant 65535 : i32
    %c65535_i32_4 = arith.constant 65535 : i32
    %5 = arith.andi %4, %c65535_i32_4 : i32
    %alloc_5 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %5, %alloc_5[] {to = "nouts"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "prog"} : memref<16xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<16xi32>)
    %6 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %7 = arith.index_cast %c0_i32_8 : i32 to index
    %8 = arith.index_cast %6 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %9 = arith.index_cast %c1_i32_9 : i32 to index
    scf.for %arg5 = %7 to %8 step %9 {
      %28 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_41 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %28, %alloc_41[] {to = "word"} : memref<i32>
      %29 = affine.load %alloc_41[] {from = "word"} : memref<i32>
      memref.store %29, %alloc_6[%arg5] {to = "prog"} : memref<16xi32>
    } {loop_name = "pc", op_name = "S_pc_0"}
    %10 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c16_i32_10 = arith.constant 16 : i32
    %c16_i32_11 = arith.constant 16 : i32
    %11 = arith.extsi %c16_i32_11 : i32 to i33
    %12 = arith.extsi %10 : i32 to i33
    %13 = arith.subi %11, %12 : i33
    %c0_i32_12 = arith.constant 0 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %14 = arith.index_cast %c0_i32_13 : i32 to index
    %15 = arith.index_cast %13 : i33 to index
    %c1_i32_14 = arith.constant 1 : i32
    %c1_i32_15 = arith.constant 1 : i32
    %16 = arith.index_cast %c1_i32_15 : i32 to index
    scf.for %arg5 = %14 to %15 step %16 {
      %28 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_41 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %28, %alloc_41[] {to = "spare"} : memref<i32>
    } {loop_name = "_pad", op_name = "S__pad_1"}
    %17 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32>
    %alloc_16 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %17, %alloc_16[] {to = "denom"} : memref<i32>
    %c0_i32_17 = arith.constant 0 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %alloc_19 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32_18, %alloc_19[] {to = "rcp"} : memref<i32>
    %18 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
    %c0_i32_20 = arith.constant 0 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %19 = arith.cmpi sgt, %18, %c0_i32_21 : i32
    scf.if %19 {
      %c1_i32_41 = arith.constant 1 : i32
      %c1_i32_42 = arith.constant 1 : i32
      %c14_i32 = arith.constant 14 : i32
      %c14_i32_43 = arith.constant 14 : i32
      %28 = arith.shli %c1_i32_42, %c14_i32_43 : i32
      %29 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
      %30 = arith.floordivsi %28, %29 : i32
      affine.store %30, %alloc_19[] {to = "rcp"} : memref<i32>
    }
    %c0_i32_22 = arith.constant 0 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %alloc_24 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32_23, %alloc_24[] {to = "r0"} : memref<i32>
    %c0_i32_25 = arith.constant 0 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %alloc_27 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32_26, %alloc_27[] {to = "r1"} : memref<i32>
    %c0_i32_28 = arith.constant 0 : i32
    %c0_i32_29 = arith.constant 0 : i32
    %alloc_30 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32_29, %alloc_30[] {to = "r2"} : memref<i32>
    %c0_i32_31 = arith.constant 0 : i32
    %c0_i32_32 = arith.constant 0 : i32
    %alloc_33 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32_32, %alloc_33[] {to = "r3"} : memref<i32>
    %c0_i32_34 = arith.constant 0 : i32
    %c0_i32_35 = arith.constant 0 : i32
    %alloc_36 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32_35, %alloc_36[] {to = "pc2"} : memref<i32>
    %20 = affine.load %alloc_5[] {from = "nouts"} : memref<i32>
    %21 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %22 = arith.extsi %20 : i32 to i64
    %23 = arith.extsi %21 : i32 to i64
    %24 = arith.muli %22, %23 : i64
    %c0_i32_37 = arith.constant 0 : i32
    %c0_i32_38 = arith.constant 0 : i32
    %25 = arith.index_cast %c0_i32_38 : i32 to index
    %26 = arith.index_cast %24 : i64 to index
    %c1_i32_39 = arith.constant 1 : i32
    %c1_i32_40 = arith.constant 1 : i32
    %27 = arith.index_cast %c1_i32_40 : i32 to index
    scf.for %arg5 = %25 to %26 step %27 {
      %28 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %29 = arith.index_cast %28 : i32 to index
      %30 = memref.load %alloc_6[%29] {from = "prog"} : memref<16xi32>
      %alloc_41 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %30, %alloc_41[] {to = "word2"} : memref<i32>
      %31 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_42 = arith.constant 24 : i32
      %32 = arith.shrsi %31, %c24_i32_42 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_43 = arith.constant 255 : i32
      %33 = arith.andi %32, %c255_i32_43 : i32
      %alloc_44 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %33, %alloc_44[] {to = "opcode"} : memref<i32>
      %34 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c20_i32 = arith.constant 20 : i32
      %c20_i32_45 = arith.constant 20 : i32
      %35 = arith.shrsi %34, %c20_i32_45 : i32
      %c15_i32 = arith.constant 15 : i32
      %c15_i32_46 = arith.constant 15 : i32
      %36 = arith.andi %35, %c15_i32_46 : i32
      %alloc_47 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %36, %alloc_47[] {to = "dst"} : memref<i32>
      %37 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c16_i32_48 = arith.constant 16 : i32
      %c16_i32_49 = arith.constant 16 : i32
      %38 = arith.shrsi %37, %c16_i32_49 : i32
      %c15_i32_50 = arith.constant 15 : i32
      %c15_i32_51 = arith.constant 15 : i32
      %39 = arith.andi %38, %c15_i32_51 : i32
      %alloc_52 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %39, %alloc_52[] {to = "src"} : memref<i32>
      %40 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c65535_i32_53 = arith.constant 65535 : i32
      %c65535_i32_54 = arith.constant 65535 : i32
      %41 = arith.andi %40, %c65535_i32_54 : i32
      %alloc_55 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %41, %alloc_55[] {to = "imm"} : memref<i32>
      %42 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_56 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %42, %alloc_56[] {to = "d"} : memref<i32>
      %43 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_58 = arith.constant 1 : i32
      %44 = arith.cmpi eq, %43, %c1_i32_58 : i32
      scf.if %44 {
        %60 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %60, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %60 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %61 = arith.cmpi eq, %60, %c2_i32_70 : i32
        scf.if %61 {
          %62 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %62, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %62 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %63 = arith.cmpi eq, %62, %c3_i32_71 : i32
          scf.if %63 {
            %64 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %64, %alloc_56[] {to = "d"} : memref<i32>
          }
        }
      }
      %45 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_59 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %45, %alloc_59[] {to = "a"} : memref<i32>
      %46 = affine.load %alloc_52[] {from = "src"} : memref<i32>
      %c1_i32_60 = arith.constant 1 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %47 = arith.cmpi eq, %46, %c1_i32_61 : i32
      scf.if %47 {
        %60 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %60, %alloc_59[] {to = "a"} : memref<i32>
      } else {
        %60 = affine.load %alloc_52[] {from = "src"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %61 = arith.cmpi eq, %60, %c2_i32_70 : i32
        scf.if %61 {
          %62 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %62, %alloc_59[] {to = "a"} : memref<i32>
        } else {
          %62 = affine.load %alloc_52[] {from = "src"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %63 = arith.cmpi eq, %62, %c3_i32_71 : i32
          scf.if %63 {
            %64 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %64, %alloc_59[] {to = "a"} : memref<i32>
          }
        }
      }
      %c1_i32_62 = arith.constant 1 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %alloc_64 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32_63, %alloc_64[] {to = "wr"} : memref<i32>
      %48 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
      %c9_i32 = arith.constant 9 : i32
      %c9_i32_65 = arith.constant 9 : i32
      %49 = arith.cmpi eq, %48, %c9_i32_65 : i32
      scf.if %49 {
        %60 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
        %alloc_70 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %60, %alloc_70[] {to = "zz"} : memref<i32>
        %61 = affine.load %alloc_56[] {from = "d"} : memref<i32>
        %62 = affine.load %alloc_70[] {from = "zz"} : memref<i32>
        %63 = arith.extsi %61 : i32 to i33
        %64 = arith.extsi %62 : i32 to i33
        %65 = arith.addi %63, %64 : i33
        %66 = arith.trunci %65 : i33 to i32
        affine.store %66, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %60 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
        %c1_i32_70 = arith.constant 1 : i32
        %c1_i32_71 = arith.constant 1 : i32
        %61 = arith.cmpi eq, %60, %c1_i32_71 : i32
        scf.if %61 {
          %62 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
          %alloc_72 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %62, %alloc_72[] {to = "z2"} : memref<i32>
          %63 = affine.load %alloc_72[] {from = "z2"} : memref<i32>
          affine.store %63, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %62 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_72 = arith.constant 2 : i32
          %63 = arith.cmpi eq, %62, %c2_i32_72 : i32
          scf.if %63 {
            %64 = affine.load %alloc_52[] {from = "src"} : memref<i32>
            %65 = arith.index_cast %64 : i32 to index
            %66 = memref.load %arg0[%arg1, %65] {from = "local_Bias"} : memref<4x2xi32>
            affine.store %66, %alloc_56[] {to = "d"} : memref<i32>
          } else {
            %64 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_73 = arith.constant 3 : i32
            %65 = arith.cmpi eq, %64, %c3_i32_73 : i32
            scf.if %65 {
              %66 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
              affine.store %66, %alloc_56[] {to = "d"} : memref<i32>
            } else {
              %66 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_74 = arith.constant 4 : i32
              %67 = arith.cmpi eq, %66, %c4_i32_74 : i32
              scf.if %67 {
                %68 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                %69 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                %70 = arith.extsi %68 : i32 to i33
                %71 = arith.extsi %69 : i32 to i33
                %72 = arith.addi %70, %71 : i33
                %73 = arith.trunci %72 : i33 to i32
                affine.store %73, %alloc_56[] {to = "d"} : memref<i32>
              } else {
                %68 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_75 = arith.constant 5 : i32
                %69 = arith.cmpi eq, %68, %c5_i32_75 : i32
                scf.if %69 {
                  %70 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                  %71 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                  %72 = arith.extsi %70 : i32 to i64
                  %73 = arith.extsi %71 : i32 to i64
                  %74 = arith.muli %72, %73 : i64
                  %75 = arith.trunci %74 : i64 to i32
                  affine.store %75, %alloc_56[] {to = "d"} : memref<i32>
                } else {
                  %70 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_76 = arith.constant 6 : i32
                  %71 = arith.cmpi eq, %70, %c6_i32_76 : i32
                  scf.if %71 {
                    %72 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                    %73 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                    %74 = arith.cmpi sgt, %72, %73 : i32
                    scf.if %74 {
                      %75 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                      affine.store %75, %alloc_56[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %72 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_77 = arith.constant 7 : i32
                    %73 = arith.cmpi eq, %72, %c7_i32_77 : i32
                    scf.if %73 {
                      %74 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                      %75 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
                      %76 = arith.shrsi %74, %75 : i32
                      affine.store %76, %alloc_56[] {to = "d"} : memref<i32>
                    } else {
                      %74 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                      %c10_i32 = arith.constant 10 : i32
                      %c10_i32_78 = arith.constant 10 : i32
                      %75 = arith.cmpi eq, %74, %c10_i32_78 : i32
                      scf.if %75 {
                        %76 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                        %77 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                        %78 = arith.extsi %76 : i32 to i33
                        %79 = arith.extsi %77 : i32 to i33
                        %80 = arith.subi %78, %79 : i33
                        %81 = arith.trunci %80 : i33 to i32
                        affine.store %81, %alloc_56[] {to = "d"} : memref<i32>
                      } else {
                        %76 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                        %c11_i32 = arith.constant 11 : i32
                        %c11_i32_79 = arith.constant 11 : i32
                        %77 = arith.cmpi eq, %76, %c11_i32_79 : i32
                        scf.if %77 {
                          %78 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                          %alloc_80 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %78, %alloc_80[] {to = "e"} : memref<i32>
                          %79 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c0_i32_81 = arith.constant 0 : i32
                          %c0_i32_82 = arith.constant 0 : i32
                          %80 = arith.cmpi slt, %79, %c0_i32_82 : i32
                          scf.if %80 {
                            %c0_i32_86 = arith.constant 0 : i32
                            %c0_i32_87 = arith.constant 0 : i32
                            affine.store %c0_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %81 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c30_i32 = arith.constant 30 : i32
                          %c30_i32_83 = arith.constant 30 : i32
                          %82 = arith.cmpi sgt, %81, %c30_i32_83 : i32
                          scf.if %82 {
                            %c30_i32_86 = arith.constant 30 : i32
                            %c30_i32_87 = arith.constant 30 : i32
                            affine.store %c30_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %83 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c1_i32_84 = arith.constant 1 : i32
                          %c1_i32_85 = arith.constant 1 : i32
                          %84 = arith.shli %c1_i32_85, %83 : i32
                          affine.store %84, %alloc_56[] {to = "d"} : memref<i32>
                        } else {
                          %78 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                          %c12_i32 = arith.constant 12 : i32
                          %c12_i32_80 = arith.constant 12 : i32
                          %79 = arith.cmpi eq, %78, %c12_i32_80 : i32
                          scf.if %79 {
                            %80 = affine.load %alloc_19[] {from = "rcp"} : memref<i32>
                            affine.store %80, %alloc_56[] {to = "d"} : memref<i32>
                          } else {
                            %80 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                            %c8_i32 = arith.constant 8 : i32
                            %c8_i32_81 = arith.constant 8 : i32
                            %81 = arith.cmpi eq, %80, %c8_i32_81 : i32
                            scf.if %81 {
                              %82 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg3, [], %82) : !allo.stream<i32, 4> contains i32
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
                            } else {
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
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
      %50 = affine.load %alloc_64[] {from = "wr"} : memref<i32>
      %c1_i32_66 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      %51 = arith.cmpi eq, %50, %c1_i32_67 : i32
      scf.if %51 {
        %60 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        %61 = arith.cmpi eq, %60, %c0_i32_71 : i32
        scf.if %61 {
          %62 = affine.load %alloc_56[] {from = "d"} : memref<i32>
          affine.store %62, %alloc_24[] {to = "r0"} : memref<i32>
        } else {
          %62 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c1_i32_72 = arith.constant 1 : i32
          %c1_i32_73 = arith.constant 1 : i32
          %63 = arith.cmpi eq, %62, %c1_i32_73 : i32
          scf.if %63 {
            %64 = affine.load %alloc_56[] {from = "d"} : memref<i32>
            affine.store %64, %alloc_27[] {to = "r1"} : memref<i32>
          } else {
            %64 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_74 = arith.constant 2 : i32
            %65 = arith.cmpi eq, %64, %c2_i32_74 : i32
            scf.if %65 {
              %66 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %66, %alloc_30[] {to = "r2"} : memref<i32>
            } else {
              %66 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %66, %alloc_33[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %52 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %53 = arith.extsi %52 : i32 to i33
      %c1_i32_68 = arith.constant 1 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %54 = arith.extsi %c1_i32_69 : i32 to i33
      %55 = arith.addi %53, %54 : i33
      %56 = arith.trunci %55 : i33 to i32
      affine.store %56, %alloc_36[] {to = "pc2"} : memref<i32>
      %57 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %58 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
      %59 = arith.cmpi eq, %57, %58 : i32
      scf.if %59 {
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        affine.store %c0_i32_71, %alloc_36[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2"}
    return
  }
  func.func @vpu_r2(%arg0: memref<4x2xi32>, %arg1: index, %arg2: !allo.stream<i32, 17>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 4>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 17> -> i32
    %alloc = memref.alloc() {name = "header"} : memref<i32>
    affine.store %0, %alloc[] {to = "header"} : memref<i32>
    %1 = affine.load %alloc[] {from = "header"} : memref<i32>
    allo.stream_put(%arg3, [], %1) : !allo.stream<i32, 2> contains i32
    %2 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c65535_i32 = arith.constant 65535 : i32
    %c65535_i32_0 = arith.constant 65535 : i32
    %3 = arith.andi %2, %c65535_i32_0 : i32
    %alloc_1 = memref.alloc() {name = "plen"} : memref<i32>
    affine.store %3, %alloc_1[] {to = "plen"} : memref<i32>
    %4 = affine.load %alloc[] {from = "header"} : memref<i32>
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_2 = arith.constant 16 : i32
    %5 = arith.shrsi %4, %c16_i32_2 : i32
    %c65535_i32_3 = arith.constant 65535 : i32
    %c65535_i32_4 = arith.constant 65535 : i32
    %6 = arith.andi %5, %c65535_i32_4 : i32
    %alloc_5 = memref.alloc() {name = "nouts"} : memref<i32>
    affine.store %6, %alloc_5[] {to = "nouts"} : memref<i32>
    %alloc_6 = memref.alloc() {name = "prog"} : memref<16xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<16xi32>)
    %7 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_i32_8 = arith.constant 0 : i32
    %8 = arith.index_cast %c0_i32_8 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %10 = arith.index_cast %c1_i32_9 : i32 to index
    scf.for %arg6 = %8 to %9 step %10 {
      %29 = allo.stream_get(%arg2, []) : !allo.stream<i32, 17> -> i32
      %alloc_41 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %29, %alloc_41[] {to = "word"} : memref<i32>
      %30 = affine.load %alloc_41[] {from = "word"} : memref<i32>
      memref.store %30, %alloc_6[%arg6] {to = "prog"} : memref<16xi32>
      %31 = affine.load %alloc_41[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %31) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    %11 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %c16_i32_10 = arith.constant 16 : i32
    %c16_i32_11 = arith.constant 16 : i32
    %12 = arith.extsi %c16_i32_11 : i32 to i33
    %13 = arith.extsi %11 : i32 to i33
    %14 = arith.subi %12, %13 : i33
    %c0_i32_12 = arith.constant 0 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %15 = arith.index_cast %c0_i32_13 : i32 to index
    %16 = arith.index_cast %14 : i33 to index
    %c1_i32_14 = arith.constant 1 : i32
    %c1_i32_15 = arith.constant 1 : i32
    %17 = arith.index_cast %c1_i32_15 : i32 to index
    scf.for %arg6 = %15 to %16 step %17 {
      %29 = allo.stream_get(%arg2, []) : !allo.stream<i32, 17> -> i32
      %alloc_41 = memref.alloc() {name = "spare"} : memref<i32>
      affine.store %29, %alloc_41[] {to = "spare"} : memref<i32>
      %30 = affine.load %alloc_41[] {from = "spare"} : memref<i32>
      allo.stream_put(%arg3, [], %30) : !allo.stream<i32, 2> contains i32
    } {loop_name = "_pad", op_name = "S__pad_1"}
    %18 = affine.load %arg0[%arg1, 1] {from = "local_Bias"} : memref<4x2xi32>
    %alloc_16 = memref.alloc() {name = "denom"} : memref<i32>
    affine.store %18, %alloc_16[] {to = "denom"} : memref<i32>
    %c0_i32_17 = arith.constant 0 : i32
    %c0_i32_18 = arith.constant 0 : i32
    %alloc_19 = memref.alloc() {name = "rcp"} : memref<i32>
    affine.store %c0_i32_18, %alloc_19[] {to = "rcp"} : memref<i32>
    %19 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
    %c0_i32_20 = arith.constant 0 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %20 = arith.cmpi sgt, %19, %c0_i32_21 : i32
    scf.if %20 {
      %c1_i32_41 = arith.constant 1 : i32
      %c1_i32_42 = arith.constant 1 : i32
      %c14_i32 = arith.constant 14 : i32
      %c14_i32_43 = arith.constant 14 : i32
      %29 = arith.shli %c1_i32_42, %c14_i32_43 : i32
      %30 = affine.load %alloc_16[] {from = "denom"} : memref<i32>
      %31 = arith.floordivsi %29, %30 : i32
      affine.store %31, %alloc_19[] {to = "rcp"} : memref<i32>
    }
    %c0_i32_22 = arith.constant 0 : i32
    %c0_i32_23 = arith.constant 0 : i32
    %alloc_24 = memref.alloc() {name = "r0"} : memref<i32>
    affine.store %c0_i32_23, %alloc_24[] {to = "r0"} : memref<i32>
    %c0_i32_25 = arith.constant 0 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %alloc_27 = memref.alloc() {name = "r1"} : memref<i32>
    affine.store %c0_i32_26, %alloc_27[] {to = "r1"} : memref<i32>
    %c0_i32_28 = arith.constant 0 : i32
    %c0_i32_29 = arith.constant 0 : i32
    %alloc_30 = memref.alloc() {name = "r2"} : memref<i32>
    affine.store %c0_i32_29, %alloc_30[] {to = "r2"} : memref<i32>
    %c0_i32_31 = arith.constant 0 : i32
    %c0_i32_32 = arith.constant 0 : i32
    %alloc_33 = memref.alloc() {name = "r3"} : memref<i32>
    affine.store %c0_i32_32, %alloc_33[] {to = "r3"} : memref<i32>
    %c0_i32_34 = arith.constant 0 : i32
    %c0_i32_35 = arith.constant 0 : i32
    %alloc_36 = memref.alloc() {name = "pc2"} : memref<i32>
    affine.store %c0_i32_35, %alloc_36[] {to = "pc2"} : memref<i32>
    %21 = affine.load %alloc_5[] {from = "nouts"} : memref<i32>
    %22 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
    %23 = arith.extsi %21 : i32 to i64
    %24 = arith.extsi %22 : i32 to i64
    %25 = arith.muli %23, %24 : i64
    %c0_i32_37 = arith.constant 0 : i32
    %c0_i32_38 = arith.constant 0 : i32
    %26 = arith.index_cast %c0_i32_38 : i32 to index
    %27 = arith.index_cast %25 : i64 to index
    %c1_i32_39 = arith.constant 1 : i32
    %c1_i32_40 = arith.constant 1 : i32
    %28 = arith.index_cast %c1_i32_40 : i32 to index
    scf.for %arg6 = %26 to %27 step %28 {
      %29 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %alloc_6[%30] {from = "prog"} : memref<16xi32>
      %alloc_41 = memref.alloc() {name = "word2"} : memref<i32>
      affine.store %31, %alloc_41[] {to = "word2"} : memref<i32>
      %32 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c24_i32 = arith.constant 24 : i32
      %c24_i32_42 = arith.constant 24 : i32
      %33 = arith.shrsi %32, %c24_i32_42 : i32
      %c255_i32 = arith.constant 255 : i32
      %c255_i32_43 = arith.constant 255 : i32
      %34 = arith.andi %33, %c255_i32_43 : i32
      %alloc_44 = memref.alloc() {name = "opcode"} : memref<i32>
      affine.store %34, %alloc_44[] {to = "opcode"} : memref<i32>
      %35 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c20_i32 = arith.constant 20 : i32
      %c20_i32_45 = arith.constant 20 : i32
      %36 = arith.shrsi %35, %c20_i32_45 : i32
      %c15_i32 = arith.constant 15 : i32
      %c15_i32_46 = arith.constant 15 : i32
      %37 = arith.andi %36, %c15_i32_46 : i32
      %alloc_47 = memref.alloc() {name = "dst"} : memref<i32>
      affine.store %37, %alloc_47[] {to = "dst"} : memref<i32>
      %38 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c16_i32_48 = arith.constant 16 : i32
      %c16_i32_49 = arith.constant 16 : i32
      %39 = arith.shrsi %38, %c16_i32_49 : i32
      %c15_i32_50 = arith.constant 15 : i32
      %c15_i32_51 = arith.constant 15 : i32
      %40 = arith.andi %39, %c15_i32_51 : i32
      %alloc_52 = memref.alloc() {name = "src"} : memref<i32>
      affine.store %40, %alloc_52[] {to = "src"} : memref<i32>
      %41 = affine.load %alloc_41[] {from = "word2"} : memref<i32>
      %c65535_i32_53 = arith.constant 65535 : i32
      %c65535_i32_54 = arith.constant 65535 : i32
      %42 = arith.andi %41, %c65535_i32_54 : i32
      %alloc_55 = memref.alloc() {name = "imm"} : memref<i32>
      affine.store %42, %alloc_55[] {to = "imm"} : memref<i32>
      %43 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_56 = memref.alloc() {name = "d"} : memref<i32>
      affine.store %43, %alloc_56[] {to = "d"} : memref<i32>
      %44 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_58 = arith.constant 1 : i32
      %45 = arith.cmpi eq, %44, %c1_i32_58 : i32
      scf.if %45 {
        %61 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %61, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %61 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %62 = arith.cmpi eq, %61, %c2_i32_70 : i32
        scf.if %62 {
          %63 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %63, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %63 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %64 = arith.cmpi eq, %63, %c3_i32_71 : i32
          scf.if %64 {
            %65 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %65, %alloc_56[] {to = "d"} : memref<i32>
          }
        }
      }
      %46 = affine.load %alloc_24[] {from = "r0"} : memref<i32>
      %alloc_59 = memref.alloc() {name = "a"} : memref<i32>
      affine.store %46, %alloc_59[] {to = "a"} : memref<i32>
      %47 = affine.load %alloc_52[] {from = "src"} : memref<i32>
      %c1_i32_60 = arith.constant 1 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %48 = arith.cmpi eq, %47, %c1_i32_61 : i32
      scf.if %48 {
        %61 = affine.load %alloc_27[] {from = "r1"} : memref<i32>
        affine.store %61, %alloc_59[] {to = "a"} : memref<i32>
      } else {
        %61 = affine.load %alloc_52[] {from = "src"} : memref<i32>
        %c2_i32 = arith.constant 2 : i32
        %c2_i32_70 = arith.constant 2 : i32
        %62 = arith.cmpi eq, %61, %c2_i32_70 : i32
        scf.if %62 {
          %63 = affine.load %alloc_30[] {from = "r2"} : memref<i32>
          affine.store %63, %alloc_59[] {to = "a"} : memref<i32>
        } else {
          %63 = affine.load %alloc_52[] {from = "src"} : memref<i32>
          %c3_i32 = arith.constant 3 : i32
          %c3_i32_71 = arith.constant 3 : i32
          %64 = arith.cmpi eq, %63, %c3_i32_71 : i32
          scf.if %64 {
            %65 = affine.load %alloc_33[] {from = "r3"} : memref<i32>
            affine.store %65, %alloc_59[] {to = "a"} : memref<i32>
          }
        }
      }
      %c1_i32_62 = arith.constant 1 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %alloc_64 = memref.alloc() {name = "wr"} : memref<i32>
      affine.store %c1_i32_63, %alloc_64[] {to = "wr"} : memref<i32>
      %49 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
      %c9_i32 = arith.constant 9 : i32
      %c9_i32_65 = arith.constant 9 : i32
      %50 = arith.cmpi eq, %49, %c9_i32_65 : i32
      scf.if %50 {
        %61 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_70 = memref.alloc() {name = "zz"} : memref<i32>
        affine.store %61, %alloc_70[] {to = "zz"} : memref<i32>
        %62 = affine.load %alloc_56[] {from = "d"} : memref<i32>
        %63 = affine.load %alloc_70[] {from = "zz"} : memref<i32>
        %64 = arith.extsi %62 : i32 to i33
        %65 = arith.extsi %63 : i32 to i33
        %66 = arith.addi %64, %65 : i33
        %67 = arith.trunci %66 : i33 to i32
        affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
      } else {
        %61 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
        %c1_i32_70 = arith.constant 1 : i32
        %c1_i32_71 = arith.constant 1 : i32
        %62 = arith.cmpi eq, %61, %c1_i32_71 : i32
        scf.if %62 {
          %63 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_72 = memref.alloc() {name = "z2"} : memref<i32>
          affine.store %63, %alloc_72[] {to = "z2"} : memref<i32>
          %64 = affine.load %alloc_72[] {from = "z2"} : memref<i32>
          affine.store %64, %alloc_56[] {to = "d"} : memref<i32>
        } else {
          %63 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_72 = arith.constant 2 : i32
          %64 = arith.cmpi eq, %63, %c2_i32_72 : i32
          scf.if %64 {
            %65 = affine.load %alloc_52[] {from = "src"} : memref<i32>
            %66 = arith.index_cast %65 : i32 to index
            %67 = memref.load %arg0[%arg1, %66] {from = "local_Bias"} : memref<4x2xi32>
            affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
          } else {
            %65 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_73 = arith.constant 3 : i32
            %66 = arith.cmpi eq, %65, %c3_i32_73 : i32
            scf.if %66 {
              %67 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
              affine.store %67, %alloc_56[] {to = "d"} : memref<i32>
            } else {
              %67 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_74 = arith.constant 4 : i32
              %68 = arith.cmpi eq, %67, %c4_i32_74 : i32
              scf.if %68 {
                %69 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                %70 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                %71 = arith.extsi %69 : i32 to i33
                %72 = arith.extsi %70 : i32 to i33
                %73 = arith.addi %71, %72 : i33
                %74 = arith.trunci %73 : i33 to i32
                affine.store %74, %alloc_56[] {to = "d"} : memref<i32>
              } else {
                %69 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_75 = arith.constant 5 : i32
                %70 = arith.cmpi eq, %69, %c5_i32_75 : i32
                scf.if %70 {
                  %71 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                  %72 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                  %73 = arith.extsi %71 : i32 to i64
                  %74 = arith.extsi %72 : i32 to i64
                  %75 = arith.muli %73, %74 : i64
                  %76 = arith.trunci %75 : i64 to i32
                  affine.store %76, %alloc_56[] {to = "d"} : memref<i32>
                } else {
                  %71 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_76 = arith.constant 6 : i32
                  %72 = arith.cmpi eq, %71, %c6_i32_76 : i32
                  scf.if %72 {
                    %73 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                    %74 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                    %75 = arith.cmpi sgt, %73, %74 : i32
                    scf.if %75 {
                      %76 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                      affine.store %76, %alloc_56[] {to = "d"} : memref<i32>
                    }
                  } else {
                    %73 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_77 = arith.constant 7 : i32
                    %74 = arith.cmpi eq, %73, %c7_i32_77 : i32
                    scf.if %74 {
                      %75 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                      %76 = affine.load %alloc_55[] {from = "imm"} : memref<i32>
                      %77 = arith.shrsi %75, %76 : i32
                      affine.store %77, %alloc_56[] {to = "d"} : memref<i32>
                    } else {
                      %75 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                      %c10_i32 = arith.constant 10 : i32
                      %c10_i32_78 = arith.constant 10 : i32
                      %76 = arith.cmpi eq, %75, %c10_i32_78 : i32
                      scf.if %76 {
                        %77 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                        %78 = affine.load %alloc_59[] {from = "a"} : memref<i32>
                        %79 = arith.extsi %77 : i32 to i33
                        %80 = arith.extsi %78 : i32 to i33
                        %81 = arith.subi %79, %80 : i33
                        %82 = arith.trunci %81 : i33 to i32
                        affine.store %82, %alloc_56[] {to = "d"} : memref<i32>
                      } else {
                        %77 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                        %c11_i32 = arith.constant 11 : i32
                        %c11_i32_79 = arith.constant 11 : i32
                        %78 = arith.cmpi eq, %77, %c11_i32_79 : i32
                        scf.if %78 {
                          %79 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                          %alloc_80 = memref.alloc() {name = "e"} : memref<i32>
                          affine.store %79, %alloc_80[] {to = "e"} : memref<i32>
                          %80 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c0_i32_81 = arith.constant 0 : i32
                          %c0_i32_82 = arith.constant 0 : i32
                          %81 = arith.cmpi slt, %80, %c0_i32_82 : i32
                          scf.if %81 {
                            %c0_i32_86 = arith.constant 0 : i32
                            %c0_i32_87 = arith.constant 0 : i32
                            affine.store %c0_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %82 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c30_i32 = arith.constant 30 : i32
                          %c30_i32_83 = arith.constant 30 : i32
                          %83 = arith.cmpi sgt, %82, %c30_i32_83 : i32
                          scf.if %83 {
                            %c30_i32_86 = arith.constant 30 : i32
                            %c30_i32_87 = arith.constant 30 : i32
                            affine.store %c30_i32_87, %alloc_80[] {to = "e"} : memref<i32>
                          }
                          %84 = affine.load %alloc_80[] {from = "e"} : memref<i32>
                          %c1_i32_84 = arith.constant 1 : i32
                          %c1_i32_85 = arith.constant 1 : i32
                          %85 = arith.shli %c1_i32_85, %84 : i32
                          affine.store %85, %alloc_56[] {to = "d"} : memref<i32>
                        } else {
                          %79 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                          %c12_i32 = arith.constant 12 : i32
                          %c12_i32_80 = arith.constant 12 : i32
                          %80 = arith.cmpi eq, %79, %c12_i32_80 : i32
                          scf.if %80 {
                            %81 = affine.load %alloc_19[] {from = "rcp"} : memref<i32>
                            affine.store %81, %alloc_56[] {to = "d"} : memref<i32>
                          } else {
                            %81 = affine.load %alloc_44[] {from = "opcode"} : memref<i32>
                            %c8_i32 = arith.constant 8 : i32
                            %c8_i32_81 = arith.constant 8 : i32
                            %82 = arith.cmpi eq, %81, %c8_i32_81 : i32
                            scf.if %82 {
                              %83 = affine.load %alloc_56[] {from = "d"} : memref<i32>
                              allo.stream_put(%arg4, [], %83) : !allo.stream<i32, 4> contains i32
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
                            } else {
                              %c0_i32_82 = arith.constant 0 : i32
                              %c0_i32_83 = arith.constant 0 : i32
                              affine.store %c0_i32_83, %alloc_64[] {to = "wr"} : memref<i32>
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
      %51 = affine.load %alloc_64[] {from = "wr"} : memref<i32>
      %c1_i32_66 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      %52 = arith.cmpi eq, %51, %c1_i32_67 : i32
      scf.if %52 {
        %61 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        %62 = arith.cmpi eq, %61, %c0_i32_71 : i32
        scf.if %62 {
          %63 = affine.load %alloc_56[] {from = "d"} : memref<i32>
          affine.store %63, %alloc_24[] {to = "r0"} : memref<i32>
        } else {
          %63 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
          %c1_i32_72 = arith.constant 1 : i32
          %c1_i32_73 = arith.constant 1 : i32
          %64 = arith.cmpi eq, %63, %c1_i32_73 : i32
          scf.if %64 {
            %65 = affine.load %alloc_56[] {from = "d"} : memref<i32>
            affine.store %65, %alloc_27[] {to = "r1"} : memref<i32>
          } else {
            %65 = affine.load %alloc_47[] {from = "dst"} : memref<i32>
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_74 = arith.constant 2 : i32
            %66 = arith.cmpi eq, %65, %c2_i32_74 : i32
            scf.if %66 {
              %67 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %67, %alloc_30[] {to = "r2"} : memref<i32>
            } else {
              %67 = affine.load %alloc_56[] {from = "d"} : memref<i32>
              affine.store %67, %alloc_33[] {to = "r3"} : memref<i32>
            }
          }
        }
      }
      %53 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %54 = arith.extsi %53 : i32 to i33
      %c1_i32_68 = arith.constant 1 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %55 = arith.extsi %c1_i32_69 : i32 to i33
      %56 = arith.addi %54, %55 : i33
      %57 = arith.trunci %56 : i33 to i32
      affine.store %57, %alloc_36[] {to = "pc2"} : memref<i32>
      %58 = affine.load %alloc_36[] {from = "pc2"} : memref<i32>
      %59 = affine.load %alloc_1[] {from = "plen"} : memref<i32>
      %60 = arith.cmpi eq, %58, %59 : i32
      scf.if %60 {
        %c0_i32_70 = arith.constant 0 : i32
        %c0_i32_71 = arith.constant 0 : i32
        affine.store %c0_i32_71, %alloc_36[] {to = "pc2"} : memref<i32>
      }
    } {loop_name = "_k", op_name = "S__k_2"}
    return
  }
  func.func @vpu_y_out_drain(%arg0: memref<4x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 4>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 4> -> i32
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<4x4xi32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<4x4xi8>, %arg1: memref<5x4xi32>, %arg2: memref<17xi32>, %arg3: memref<4x4x4xi8>, %arg4: memref<4x2xi32>, %arg5: memref<4x4xi32>) attributes {dataflow, itypes = "ssssss", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [4], families = [#spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 4, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "mac_a_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @mac_a_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<4x4xi8>
    spmw.map(%arg1) topology = <grid = [4], families = [#spmw.family<name = "mac_op_in_bind", type = i32, block = [], depth = 5, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "mac_op_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @mac_op_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<5x4xi32>
    spmw.map(%arg2) topology = <grid = [1], families = [#spmw.family<name = "vpu_op_in_bind", type = i32, block = [], depth = 17, shape = [1]>], ports = [#spmw.port_map<port = "chan", family = "vpu_op_in_bind", kind = "table", slots = dense<0> : tensor<1xi32>>]> roles = [#spmw.role<unit = @vpu_op_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<1xi32> : memref<17xi32>
    spmw.map(%arg3) topology = <grid = [4, 4], families = [#spmw.family<name = "mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_op_out_op_in", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 4, shape = [4]>, #spmw.family<name = "mac_op_in_bind", type = i32, block = [], depth = 5, shape = [4]>, #spmw.family<name = "vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "a_in", family = "mac_a_in_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "op_in", family = "mac_op_in_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "vpu_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "mac_a_out_a_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "op_in", family = "mac_op_out_op_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "op_out", family = "mac_op_out_op_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "p_in", family = "mac_p_out_p_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "p_out", family = "mac_p_out_p_in", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @mac_r0, missing = [], ports = ["a_in", "a_out", "op_in", "op_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r1, missing = ["p_out"], ports = ["a_in", "a_out", "op_in", "op_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r2, missing = ["p_in"], ports = ["a_in", "a_out", "op_in", "op_out", "p_out"]>, #spmw.role<unit = @mac_r3, missing = ["a_out", "op_out"], ports = ["a_in", "op_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r4, missing = ["a_in", "op_in"], ports = ["a_in", "a_out", "op_in", "op_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r5, missing = ["a_out", "op_out", "p_out"], ports = ["a_in", "op_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r6, missing = ["a_out", "op_out", "p_in"], ports = ["a_in", "op_in", "p_out"]>, #spmw.role<unit = @mac_r7, missing = ["a_in", "op_in", "p_out"], ports = ["a_in", "a_out", "op_in", "op_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r8, missing = ["a_in", "op_in", "p_in"], ports = ["a_in", "a_out", "op_in", "op_out", "p_out"]>] classes = dense<[8, 2, 2, 6, 4, 0, 0, 3, 4, 0, 0, 3, 7, 1, 1, 5]> : tensor<16xi32> : memref<4x4x4xi8>
    spmw.map(%arg4) topology = <grid = [4], families = [#spmw.family<name = "vpu_op_out_op_in", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "vpu_op_in_bind", type = i32, block = [], depth = 17, shape = [1]>, #spmw.family<name = "vpu_y_out_bind", type = i32, block = [], depth = 4, shape = [4]>], ports = [#spmw.port_map<port = "p_out", family = "vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<4xi32>>, #spmw.port_map<port = "z_in", family = "vpu_z_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "vpu_op_in_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "y_out", family = "vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "vpu_op_out_op_in", kind = "affine", offset = [0]>, #spmw.port_map<port = "op_out", family = "vpu_op_out_op_in", kind = "affine", offset = [1]>]> roles = [#spmw.role<unit = @vpu_r0, missing = ["y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>, #spmw.role<unit = @vpu_r1, missing = ["op_out", "y_out", "z_in"], ports = ["op_in", "y_out", "z_in"]>, #spmw.role<unit = @vpu_r2, missing = ["op_in", "y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>] classes = dense<[2, 0, 0, 1]> : tensor<4xi32> : memref<4x2xi32>
    spmw.map(%arg5) topology = <grid = [4], families = [#spmw.family<name = "vpu_y_out_bind", type = i32, block = [], depth = 4, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @vpu_y_out_drain, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<4x4xi32>
    return
  }
}
