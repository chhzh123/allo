module {
  func.func @mac_a_in_load(%arg0: memref<6x4xi8>, %arg1: index, %arg2: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<6x4xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 6> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @vpu_op_in_load(%arg0: memref<8xi32>, %arg1: index, %arg2: !allo.stream<i32, 8>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 8 {
      %0 = affine.load %arg0[%arg3] {from = "local_Prog"} : memref<8xi32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 8> contains i32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @mac_r0(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r1(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r2(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg5, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r3(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg5, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r4(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r5(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg5, [], %10) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r6(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg4, [], %9) : !allo.stream<i32, 2> contains i33
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r7(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.extsi %4 : i8 to i16
      %7 = arith.muli %5, %6 : i16
      %8 = arith.extsi %2 : i32 to i33
      %9 = arith.extsi %7 : i16 to i33
      %10 = arith.addi %8, %9 : i33
      allo.stream_put(%arg6, [], %10) : !allo.stream<i32, 2> contains i33
      %11 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %11) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @mac_r8(%arg0: memref<4x4xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
      %1 = affine.load %alloc_1[] {from = "p"} : memref<i32>
      %2 = affine.load %alloc[] {from = "a"} : memref<i8>
      %3 = affine.load %arg0[%arg1, %arg2] {from = "local_W"} : memref<4x4xi8>
      %4 = arith.extsi %2 : i8 to i16
      %5 = arith.extsi %3 : i8 to i16
      %6 = arith.muli %4, %5 : i16
      %7 = arith.extsi %1 : i32 to i33
      %8 = arith.extsi %6 : i16 to i33
      %9 = arith.addi %7, %8 : i33
      allo.stream_put(%arg5, [], %9) : !allo.stream<i32, 2> contains i33
      %10 = affine.load %alloc[] {from = "a"} : memref<i8>
      allo.stream_put(%arg4, [], %10) : !allo.stream<i8, 2> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @vpu_r0(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
    affine.for %arg6 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_2 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_2 : i32) outs(%alloc_1 : memref<4xi32>)
      affine.for %arg7 = 0 to 8 {
        %1 = affine.load %alloc[%arg7] {from = "prog"} : memref<8xi32>
        %alloc_3 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_3[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_4 = arith.constant 24 : i32
        %3 = arith.shrsi %2, %c24_i32_4 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_5 = arith.constant 255 : i32
        %4 = arith.andi %3, %c255_i32_5 : i32
        %alloc_6 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_6[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_7 = arith.constant 20 : i32
        %6 = arith.shrsi %5, %c20_i32_7 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_8 = arith.constant 15 : i32
        %7 = arith.andi %6, %c15_i32_8 : i32
        %alloc_9 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_9[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_10 = arith.constant 16 : i32
        %9 = arith.shrsi %8, %c16_i32_10 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %c15_i32_12 = arith.constant 15 : i32
        %10 = arith.andi %9, %c15_i32_12 : i32
        %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_13[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_14 = arith.constant 65535 : i32
        %12 = arith.andi %11, %c65535_i32_14 : i32
        %alloc_15 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_15[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %14 = arith.cmpi eq, %13, %c1_i32_16 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_17 = arith.constant 2 : i32
          %16 = arith.cmpi eq, %15, %c2_i32_17 : i32
          scf.if %16 {
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %18 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_18 = arith.constant 3 : i32
            %18 = arith.cmpi eq, %17, %c3_i32_18 : i32
            scf.if %18 {
              %19 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_19 = arith.constant 4 : i32
              %20 = arith.cmpi eq, %19, %c4_i32_19 : i32
              scf.if %20 {
                %21 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                %31 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %32 = arith.index_cast %31 : i32 to index
                memref.store %30, %alloc_1[%32] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_20 = arith.constant 5 : i32
                %22 = arith.cmpi eq, %21, %c5_i32_20 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  %33 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %34 = arith.index_cast %33 : i32 to index
                  memref.store %32, %alloc_1[%34] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_21 = arith.constant 6 : i32
                  %24 = arith.cmpi eq, %23, %c6_i32_21 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_22 = arith.constant 7 : i32
                    %26 = arith.cmpi eq, %25, %c7_i32_22 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      %32 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      memref.store %31, %alloc_1[%33] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_23 = arith.constant 8 : i32
                      %28 = arith.cmpi eq, %27, %c8_i32_23 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
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
      } {loop_name = "step", op_name = "S_step_1"}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_r1(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 6>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
    affine.for %arg5 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<8xi32>
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg5 = 0 to 6 {
      %0 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_2 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_2 : i32) outs(%alloc_1 : memref<4xi32>)
      affine.for %arg6 = 0 to 8 {
        %1 = affine.load %alloc[%arg6] {from = "prog"} : memref<8xi32>
        %alloc_3 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_3[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_4 = arith.constant 24 : i32
        %3 = arith.shrsi %2, %c24_i32_4 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_5 = arith.constant 255 : i32
        %4 = arith.andi %3, %c255_i32_5 : i32
        %alloc_6 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_6[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_7 = arith.constant 20 : i32
        %6 = arith.shrsi %5, %c20_i32_7 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_8 = arith.constant 15 : i32
        %7 = arith.andi %6, %c15_i32_8 : i32
        %alloc_9 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_9[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_10 = arith.constant 16 : i32
        %9 = arith.shrsi %8, %c16_i32_10 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %c15_i32_12 = arith.constant 15 : i32
        %10 = arith.andi %9, %c15_i32_12 : i32
        %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_13[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_14 = arith.constant 65535 : i32
        %12 = arith.andi %11, %c65535_i32_14 : i32
        %alloc_15 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_15[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %14 = arith.cmpi eq, %13, %c1_i32_16 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_17 = arith.constant 2 : i32
          %16 = arith.cmpi eq, %15, %c2_i32_17 : i32
          scf.if %16 {
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %18 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_18 = arith.constant 3 : i32
            %18 = arith.cmpi eq, %17, %c3_i32_18 : i32
            scf.if %18 {
              %19 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_19 = arith.constant 4 : i32
              %20 = arith.cmpi eq, %19, %c4_i32_19 : i32
              scf.if %20 {
                %21 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                %31 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %32 = arith.index_cast %31 : i32 to index
                memref.store %30, %alloc_1[%32] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_20 = arith.constant 5 : i32
                %22 = arith.cmpi eq, %21, %c5_i32_20 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  %33 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %34 = arith.index_cast %33 : i32 to index
                  memref.store %32, %alloc_1[%34] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_21 = arith.constant 6 : i32
                  %24 = arith.cmpi eq, %23, %c6_i32_21 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_22 = arith.constant 7 : i32
                    %26 = arith.cmpi eq, %25, %c7_i32_22 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      %32 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      memref.store %31, %alloc_1[%33] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_23 = arith.constant 8 : i32
                      %28 = arith.cmpi eq, %27, %c8_i32_23 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
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
      } {loop_name = "step", op_name = "S_step_1"}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_r2(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 8>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<8xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<8xi32>)
    affine.for %arg6 = 0 to 8 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 8> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<8xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "z"} : memref<i32>
      %alloc_1 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_2 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_2 : i32) outs(%alloc_1 : memref<4xi32>)
      affine.for %arg7 = 0 to 8 {
        %1 = affine.load %alloc[%arg7] {from = "prog"} : memref<8xi32>
        %alloc_3 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %1, %alloc_3[] {to = "word2"} : memref<i32>
        %2 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_4 = arith.constant 24 : i32
        %3 = arith.shrsi %2, %c24_i32_4 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_5 = arith.constant 255 : i32
        %4 = arith.andi %3, %c255_i32_5 : i32
        %alloc_6 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %4, %alloc_6[] {to = "opcode"} : memref<i32>
        %5 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_7 = arith.constant 20 : i32
        %6 = arith.shrsi %5, %c20_i32_7 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_8 = arith.constant 15 : i32
        %7 = arith.andi %6, %c15_i32_8 : i32
        %alloc_9 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %7, %alloc_9[] {to = "dst"} : memref<i32>
        %8 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_10 = arith.constant 16 : i32
        %9 = arith.shrsi %8, %c16_i32_10 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %c15_i32_12 = arith.constant 15 : i32
        %10 = arith.andi %9, %c15_i32_12 : i32
        %alloc_13 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %10, %alloc_13[] {to = "src"} : memref<i32>
        %11 = affine.load %alloc_3[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_14 = arith.constant 65535 : i32
        %12 = arith.andi %11, %c65535_i32_14 : i32
        %alloc_15 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %12, %alloc_15[] {to = "imm"} : memref<i32>
        %13 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
        %c1_i32 = arith.constant 1 : i32
        %c1_i32_16 = arith.constant 1 : i32
        %14 = arith.cmpi eq, %13, %c1_i32_16 : i32
        scf.if %14 {
          %15 = affine.load %alloc_0[] {from = "z"} : memref<i32>
          %16 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
          %17 = arith.index_cast %16 : i32 to index
          memref.store %15, %alloc_1[%17] {to = "reg"} : memref<4xi32>
        } else {
          %15 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_17 = arith.constant 2 : i32
          %16 = arith.cmpi eq, %15, %c2_i32_17 : i32
          scf.if %16 {
            %17 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %18 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
            %19 = arith.index_cast %18 : i32 to index
            memref.store %17, %alloc_1[%19] {to = "reg"} : memref<4xi32>
          } else {
            %17 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_18 = arith.constant 3 : i32
            %18 = arith.cmpi eq, %17, %c3_i32_18 : i32
            scf.if %18 {
              %19 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
              %20 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
              %21 = arith.index_cast %20 : i32 to index
              memref.store %19, %alloc_1[%21] {to = "reg"} : memref<4xi32>
            } else {
              %19 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_19 = arith.constant 4 : i32
              %20 = arith.cmpi eq, %19, %c4_i32_19 : i32
              scf.if %20 {
                %21 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %22 = arith.index_cast %21 : i32 to index
                %23 = memref.load %alloc_1[%22] {from = "reg"} : memref<4xi32>
                %24 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %alloc_1[%25] {from = "reg"} : memref<4xi32>
                %27 = arith.extsi %23 : i32 to i33
                %28 = arith.extsi %26 : i32 to i33
                %29 = arith.addi %27, %28 : i33
                %30 = arith.trunci %29 : i33 to i32
                %31 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                %32 = arith.index_cast %31 : i32 to index
                memref.store %30, %alloc_1[%32] {to = "reg"} : memref<4xi32>
              } else {
                %21 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_20 = arith.constant 5 : i32
                %22 = arith.cmpi eq, %21, %c5_i32_20 : i32
                scf.if %22 {
                  %23 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %alloc_1[%24] {from = "reg"} : memref<4xi32>
                  %26 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                  %27 = arith.index_cast %26 : i32 to index
                  %28 = memref.load %alloc_1[%27] {from = "reg"} : memref<4xi32>
                  %29 = arith.extsi %25 : i32 to i64
                  %30 = arith.extsi %28 : i32 to i64
                  %31 = arith.muli %29, %30 : i64
                  %32 = arith.trunci %31 : i64 to i32
                  %33 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                  %34 = arith.index_cast %33 : i32 to index
                  memref.store %32, %alloc_1[%34] {to = "reg"} : memref<4xi32>
                } else {
                  %23 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_21 = arith.constant 6 : i32
                  %24 = arith.cmpi eq, %23, %c6_i32_21 : i32
                  scf.if %24 {
                    %25 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                    %26 = arith.index_cast %25 : i32 to index
                    %27 = memref.load %alloc_1[%26] {from = "reg"} : memref<4xi32>
                    %28 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                    %29 = arith.index_cast %28 : i32 to index
                    %30 = memref.load %alloc_1[%29] {from = "reg"} : memref<4xi32>
                    %31 = arith.cmpi sgt, %27, %30 : i32
                    scf.if %31 {
                      %32 = affine.load %alloc_13[] {from = "src"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = memref.load %alloc_1[%33] {from = "reg"} : memref<4xi32>
                      %35 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %36 = arith.index_cast %35 : i32 to index
                      memref.store %34, %alloc_1[%36] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %25 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_22 = arith.constant 7 : i32
                    %26 = arith.cmpi eq, %25, %c7_i32_22 : i32
                    scf.if %26 {
                      %27 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %28 = arith.index_cast %27 : i32 to index
                      %29 = memref.load %alloc_1[%28] {from = "reg"} : memref<4xi32>
                      %30 = affine.load %alloc_15[] {from = "imm"} : memref<i32>
                      %31 = arith.shrsi %29, %30 : i32
                      %32 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
                      %33 = arith.index_cast %32 : i32 to index
                      memref.store %31, %alloc_1[%33] {to = "reg"} : memref<4xi32>
                    } else {
                      %27 = affine.load %alloc_6[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_23 = arith.constant 8 : i32
                      %28 = arith.cmpi eq, %27, %c8_i32_23 : i32
                      scf.if %28 {
                        %29 = affine.load %alloc_9[] {from = "dst"} : memref<i32>
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
      } {loop_name = "step", op_name = "S_step_1"}
    } {loop_name = "m", op_name = "S_m_1"}
    return
  }
  func.func @vpu_y_out_drain(%arg0: memref<6x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 6> -> i32
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<6x4xi32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<6x4xi8>, %arg1: memref<8xi32>, %arg2: memref<4x4xi8>, %arg3: memref<4xi32>, %arg4: memref<6x4xi32>) attributes {dataflow, itypes = "sssss", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [4], families = [#spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 6, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "mac_a_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @mac_a_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<6x4xi8>
    spmw.map(%arg1) topology = <grid = [1], families = [#spmw.family<name = "vpu_op_in_bind", type = i32, block = [], depth = 8, shape = [1]>], ports = [#spmw.port_map<port = "chan", family = "vpu_op_in_bind", kind = "table", slots = dense<0> : tensor<1xi32>>]> roles = [#spmw.role<unit = @vpu_op_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<1xi32> : memref<8xi32>
    spmw.map(%arg2) topology = <grid = [4, 4], families = [#spmw.family<name = "mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 6, shape = [4]>, #spmw.family<name = "vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "a_in", family = "mac_a_in_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "vpu_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "mac_a_out_a_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "p_in", family = "mac_p_out_p_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "p_out", family = "mac_p_out_p_in", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @mac_r0, missing = [], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r1, missing = ["p_out"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r2, missing = ["p_in"], ports = ["a_in", "a_out", "p_out"]>, #spmw.role<unit = @mac_r3, missing = ["a_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r4, missing = ["a_in"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r5, missing = ["a_out", "p_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r6, missing = ["a_out", "p_in"], ports = ["a_in", "p_out"]>, #spmw.role<unit = @mac_r7, missing = ["a_in", "p_out"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r8, missing = ["a_in", "p_in"], ports = ["a_in", "a_out", "p_out"]>] classes = dense<[8, 2, 2, 6, 4, 0, 0, 3, 4, 0, 0, 3, 7, 1, 1, 5]> : tensor<16xi32> : memref<4x4xi8>
    spmw.map(%arg3) topology = <grid = [4], families = [#spmw.family<name = "vpu_op_out_op_in", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "vpu_op_in_bind", type = i32, block = [], depth = 8, shape = [1]>, #spmw.family<name = "vpu_y_out_bind", type = i32, block = [], depth = 6, shape = [4]>], ports = [#spmw.port_map<port = "p_out", family = "vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<4xi32>>, #spmw.port_map<port = "z_in", family = "vpu_z_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "vpu_op_in_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "y_out", family = "vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "vpu_op_out_op_in", kind = "affine", offset = [0]>, #spmw.port_map<port = "op_out", family = "vpu_op_out_op_in", kind = "affine", offset = [1]>]> roles = [#spmw.role<unit = @vpu_r0, missing = ["y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>, #spmw.role<unit = @vpu_r1, missing = ["op_out", "y_out", "z_in"], ports = ["op_in", "y_out", "z_in"]>, #spmw.role<unit = @vpu_r2, missing = ["op_in", "y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>] classes = dense<[2, 0, 0, 1]> : tensor<4xi32> : memref<4xi32>
    spmw.map(%arg4) topology = <grid = [4], families = [#spmw.family<name = "vpu_y_out_bind", type = i32, block = [], depth = 6, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @vpu_y_out_drain, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<6x4xi32>
    return
  }
}
