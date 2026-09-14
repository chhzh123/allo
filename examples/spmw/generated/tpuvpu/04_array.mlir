#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
#map1 = affine_map<(d0) -> (d0, 0)>
module {
  func.func @mac_a_in_load(%arg0: memref<6x4xi8, #map>, %arg1: index, %arg2: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<6x4xi8, #map>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 6> contains i8
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_op_in_load(%arg0: memref<8xi32, #map1>, %arg1: index, %arg2: !allo.stream<i32, 8>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 8 {
      %0 = affine.load %arg0[%arg3] {from = "local_Prog"} : memref<8xi32, #map1>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 8> contains i32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r0(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r1(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r2(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg5, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r3(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg5, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r4(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r5(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg5, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r6(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg4, [], %9) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r7(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @mac_r8(%arg0: memref<4x4xi8, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32, %alloc_0[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8, #map>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg5, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @vpu_r0(%arg0: memref<4xi32, #map1>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg6 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg6] : memref<8xi32>
    }
    affine.for %arg6 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg7 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg7] : memref<4xi32>
      }
      affine.for %arg7 = 0 to 8 {
        %1 = affine.load %alloc[%arg7] {from = "prog"} : memref<8xi32>
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
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32, #map1>
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
                        allo.stream_put(%arg4, [], %31) : !allo.stream<i32, 6> contains i32
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
  func.func @vpu_r1(%arg0: memref<4xi32, #map1>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 6>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg5 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg5] : memref<8xi32>
    }
    affine.for %arg5 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<8xi32>
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
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
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32, #map1>
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
                        allo.stream_put(%arg3, [], %31) : !allo.stream<i32, 6> contains i32
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
  func.func @vpu_r2(%arg0: memref<4xi32, #map1>, %arg1: index, %arg2: !allo.stream<i32, 8>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c20_i32 = arith.constant 20 : i32
    %c255_i32 = arith.constant 255 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    affine.for %arg6 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg6] : memref<8xi32>
    }
    affine.for %arg6 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 8> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0", pipeline_ii = 1 : ui32}
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      affine.for %arg7 = 0 to 4 {
        affine.store %c0_i32, %alloc_1[%arg7] : memref<4xi32>
      }
      affine.for %arg7 = 0 to 8 {
        %1 = affine.load %alloc[%arg7] {from = "prog"} : memref<8xi32>
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
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32, #map1>
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
                        allo.stream_put(%arg4, [], %31) : !allo.stream<i32, 6> contains i32
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
  func.func @vpu_y_out_drain(%arg0: memref<6x4xi32, #map>, %arg1: index, %arg2: !allo.stream<i32, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 6> -> i32
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<6x4xi32, #map>
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @top(%arg0: memref<6x4xi8, #map>, %arg1: memref<8xi32, #map1>, %arg2: memref<4x4xi8, #map>, %arg3: memref<4xi32, #map1>, %arg4: memref<6x4xi32, #map>) attributes {dataflow, itypes = "sssss", otypes = "", top} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %0 = allo.stream_construct() {name = "vpu_y_out_bind_3"} : !allo.stream<i32, 6>
    %1 = allo.stream_construct() {name = "vpu_y_out_bind_2"} : !allo.stream<i32, 6>
    %2 = allo.stream_construct() {name = "vpu_op_out_op_in_3"} : !allo.stream<i32, 2>
    %3 = allo.stream_construct() {name = "vpu_y_out_bind_1"} : !allo.stream<i32, 6>
    %4 = allo.stream_construct() {name = "vpu_op_out_op_in_2"} : !allo.stream<i32, 2>
    %5 = allo.stream_construct() {name = "vpu_y_out_bind_0"} : !allo.stream<i32, 6>
    %6 = allo.stream_construct() {name = "vpu_op_out_op_in_1"} : !allo.stream<i32, 2>
    %7 = allo.stream_construct() {name = "vpu_z_in_bind_3"} : !allo.stream<i32, 2>
    %8 = allo.stream_construct() {name = "vpu_z_in_bind_2"} : !allo.stream<i32, 2>
    %9 = allo.stream_construct() {name = "mac_a_out_a_in_3_3"} : !allo.stream<i8, 2>
    %10 = allo.stream_construct() {name = "vpu_z_in_bind_1"} : !allo.stream<i32, 2>
    %11 = allo.stream_construct() {name = "mac_a_out_a_in_3_2"} : !allo.stream<i8, 2>
    %12 = allo.stream_construct() {name = "vpu_z_in_bind_0"} : !allo.stream<i32, 2>
    %13 = allo.stream_construct() {name = "mac_a_out_a_in_3_1"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "mac_p_out_p_in_3_3"} : !allo.stream<i32, 2>
    %15 = allo.stream_construct() {name = "mac_p_out_p_in_3_2"} : !allo.stream<i32, 2>
    %16 = allo.stream_construct() {name = "mac_a_out_a_in_2_3"} : !allo.stream<i8, 2>
    %17 = allo.stream_construct() {name = "mac_p_out_p_in_3_1"} : !allo.stream<i32, 2>
    %18 = allo.stream_construct() {name = "mac_a_out_a_in_2_2"} : !allo.stream<i8, 2>
    %19 = allo.stream_construct() {name = "mac_p_out_p_in_3_0"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "mac_a_out_a_in_2_1"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "mac_p_out_p_in_2_3"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "mac_p_out_p_in_2_2"} : !allo.stream<i32, 2>
    %23 = allo.stream_construct() {name = "mac_a_out_a_in_1_3"} : !allo.stream<i8, 2>
    %24 = allo.stream_construct() {name = "mac_p_out_p_in_2_1"} : !allo.stream<i32, 2>
    %25 = allo.stream_construct() {name = "mac_a_out_a_in_1_2"} : !allo.stream<i8, 2>
    %26 = allo.stream_construct() {name = "mac_p_out_p_in_2_0"} : !allo.stream<i32, 2>
    %27 = allo.stream_construct() {name = "mac_a_out_a_in_1_1"} : !allo.stream<i8, 2>
    %28 = allo.stream_construct() {name = "mac_p_out_p_in_1_3"} : !allo.stream<i32, 2>
    %29 = allo.stream_construct() {name = "mac_p_out_p_in_1_2"} : !allo.stream<i32, 2>
    %30 = allo.stream_construct() {name = "mac_a_out_a_in_0_3"} : !allo.stream<i8, 2>
    %31 = allo.stream_construct() {name = "mac_p_out_p_in_1_1"} : !allo.stream<i32, 2>
    %32 = allo.stream_construct() {name = "mac_a_out_a_in_0_2"} : !allo.stream<i8, 2>
    %33 = allo.stream_construct() {name = "mac_p_out_p_in_1_0"} : !allo.stream<i32, 2>
    %34 = allo.stream_construct() {name = "mac_a_out_a_in_0_1"} : !allo.stream<i8, 2>
    %35 = allo.stream_construct() {name = "vpu_op_in_bind_0"} : !allo.stream<i32, 8>
    %36 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 6>
    %37 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 6>
    %38 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 6>
    %39 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 6>
    call @mac_a_in_load(%arg0, %c0, %39) : (memref<6x4xi8, #map>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c1, %38) : (memref<6x4xi8, #map>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c2, %37) : (memref<6x4xi8, #map>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c3, %36) : (memref<6x4xi8, #map>, index, !allo.stream<i8, 6>) -> ()
    call @vpu_op_in_load(%arg1, %c0, %35) : (memref<8xi32, #map1>, index, !allo.stream<i32, 8>) -> ()
    call @mac_r8(%arg2, %c0, %c0, %39, %34, %33) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg2, %c0, %c1, %34, %32, %31) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg2, %c0, %c2, %32, %30, %29) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r6(%arg2, %c0, %c3, %30, %28) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg2, %c1, %c0, %38, %27, %33, %26) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg2, %c1, %c1, %27, %25, %31, %24) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg2, %c1, %c2, %25, %23, %29, %22) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg2, %c1, %c3, %23, %28, %21) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg2, %c2, %c0, %37, %20, %26, %19) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg2, %c2, %c1, %20, %18, %24, %17) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg2, %c2, %c2, %18, %16, %22, %15) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg2, %c2, %c3, %16, %21, %14) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r7(%arg2, %c3, %c0, %36, %13, %19, %12) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg2, %c3, %c1, %13, %11, %17, %10) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg2, %c3, %c2, %11, %9, %15, %8) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r5(%arg2, %c3, %c3, %9, %14, %7) : (memref<4x4xi8, #map>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @vpu_r2(%arg3, %c0, %35, %6, %5, %12) : (memref<4xi32, #map1>, index, !allo.stream<i32, 8>, !allo.stream<i32, 2>, !allo.stream<i32, 6>, !allo.stream<i32, 2>) -> ()
    call @vpu_r0(%arg3, %c1, %6, %4, %3, %10) : (memref<4xi32, #map1>, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 6>, !allo.stream<i32, 2>) -> ()
    call @vpu_r0(%arg3, %c2, %4, %2, %1, %8) : (memref<4xi32, #map1>, index, !allo.stream<i32, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 6>, !allo.stream<i32, 2>) -> ()
    call @vpu_r1(%arg3, %c3, %2, %0, %7) : (memref<4xi32, #map1>, index, !allo.stream<i32, 2>, !allo.stream<i32, 6>, !allo.stream<i32, 2>) -> ()
    call @vpu_y_out_drain(%arg4, %c0, %5) : (memref<6x4xi32, #map>, index, !allo.stream<i32, 6>) -> ()
    call @vpu_y_out_drain(%arg4, %c1, %3) : (memref<6x4xi32, #map>, index, !allo.stream<i32, 6>) -> ()
    call @vpu_y_out_drain(%arg4, %c2, %1) : (memref<6x4xi32, #map>, index, !allo.stream<i32, 6>) -> ()
    call @vpu_y_out_drain(%arg4, %c3, %0) : (memref<6x4xi32, #map>, index, !allo.stream<i32, 6>) -> ()
    return
  }
}
