module {
  func.func @tiled_mac_a_in_load(%arg0: memref<12x4xi8>, %arg1: index, %arg2: !allo.stream<i8, 12>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 12 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<12x4xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 12> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @tiled_vpu_op_in_load(%arg0: memref<12xi32>, %arg1: index, %arg2: !allo.stream<i32, 12>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 12 {
      %0 = affine.load %arg0[%arg3] {from = "local_Prog"} : memref<12xi32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i32, 12> contains i32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @tiled_mac_r0(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      affine.for %arg8 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg8] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg6, [], %12) : !allo.stream<i32, 2> contains i41
        %13 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r1(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      affine.for %arg8 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg8] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg6, [], %12) : !allo.stream<i32, 2> contains i41
        %13 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r2(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      affine.for %arg7 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_0 = arith.constant 0 : i32
        %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
        %1 = affine.load %arg0[%arg1, %arg2, %arg7] {from = "local_W"} : memref<4x4x2xi8>
        %2 = arith.extsi %1 : i8 to i32
        %alloc_2 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %2, %alloc_2[] {to = "wt"} : memref<i32>
        %3 = affine.load %alloc_1[] {from = "p"} : memref<i32>
        %4 = affine.load %alloc[] {from = "a"} : memref<i8>
        %5 = affine.load %alloc_2[] {from = "wt"} : memref<i32>
        %6 = arith.extsi %4 : i8 to i40
        %7 = arith.extsi %5 : i32 to i40
        %8 = arith.muli %6, %7 : i40
        %9 = arith.extsi %3 : i32 to i41
        %10 = arith.extsi %8 : i40 to i41
        %11 = arith.addi %9, %10 : i41
        allo.stream_put(%arg5, [], %11) : !allo.stream<i32, 2> contains i41
        %12 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %12) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r3(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      affine.for %arg7 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg7] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg5, [], %12) : !allo.stream<i32, 2> contains i41
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r4(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 12>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      affine.for %arg8 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 12> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg8] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg6, [], %12) : !allo.stream<i32, 2> contains i41
        %13 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r5(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      affine.for %arg7 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg7] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg5, [], %12) : !allo.stream<i32, 2> contains i41
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r6(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    affine.for %arg5 = 0 to 6 {
      affine.for %arg6 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_0 = arith.constant 0 : i32
        %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
        %1 = affine.load %arg0[%arg1, %arg2, %arg6] {from = "local_W"} : memref<4x4x2xi8>
        %2 = arith.extsi %1 : i8 to i32
        %alloc_2 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %2, %alloc_2[] {to = "wt"} : memref<i32>
        %3 = affine.load %alloc_1[] {from = "p"} : memref<i32>
        %4 = affine.load %alloc[] {from = "a"} : memref<i8>
        %5 = affine.load %alloc_2[] {from = "wt"} : memref<i32>
        %6 = arith.extsi %4 : i8 to i40
        %7 = arith.extsi %5 : i32 to i40
        %8 = arith.muli %6, %7 : i40
        %9 = arith.extsi %3 : i32 to i41
        %10 = arith.extsi %8 : i40 to i41
        %11 = arith.addi %9, %10 : i41
        allo.stream_put(%arg4, [], %11) : !allo.stream<i32, 2> contains i41
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r7(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 12>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      affine.for %arg8 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 12> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
        %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
        %2 = affine.load %arg0[%arg1, %arg2, %arg8] {from = "local_W"} : memref<4x4x2xi8>
        %3 = arith.extsi %2 : i8 to i32
        %alloc_1 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %3, %alloc_1[] {to = "wt"} : memref<i32>
        %4 = affine.load %alloc_0[] {from = "p"} : memref<i32>
        %5 = affine.load %alloc[] {from = "a"} : memref<i8>
        %6 = affine.load %alloc_1[] {from = "wt"} : memref<i32>
        %7 = arith.extsi %5 : i8 to i40
        %8 = arith.extsi %6 : i32 to i40
        %9 = arith.muli %7, %8 : i40
        %10 = arith.extsi %4 : i32 to i41
        %11 = arith.extsi %9 : i40 to i41
        %12 = arith.addi %10, %11 : i41
        allo.stream_put(%arg6, [], %12) : !allo.stream<i32, 2> contains i41
        %13 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %13) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_mac_r8(%arg0: memref<4x4x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 12>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      affine.for %arg7 = 0 to 2 {
        %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 12> -> i8
        %alloc = memref.alloc() {name = "a"} : memref<i8>
        affine.store %0, %alloc[] {to = "a"} : memref<i8>
        %c0_i32 = arith.constant 0 : i32
        %c0_i32_0 = arith.constant 0 : i32
        %alloc_1 = memref.alloc() {name = "p"} : memref<i32>
        affine.store %c0_i32_0, %alloc_1[] {to = "p"} : memref<i32>
        %1 = affine.load %arg0[%arg1, %arg2, %arg7] {from = "local_W"} : memref<4x4x2xi8>
        %2 = arith.extsi %1 : i8 to i32
        %alloc_2 = memref.alloc() {name = "wt"} : memref<i32>
        affine.store %2, %alloc_2[] {to = "wt"} : memref<i32>
        %3 = affine.load %alloc_1[] {from = "p"} : memref<i32>
        %4 = affine.load %alloc[] {from = "a"} : memref<i8>
        %5 = affine.load %alloc_2[] {from = "wt"} : memref<i32>
        %6 = arith.extsi %4 : i8 to i40
        %7 = arith.extsi %5 : i32 to i40
        %8 = arith.muli %6, %7 : i40
        %9 = arith.extsi %3 : i32 to i41
        %10 = arith.extsi %8 : i40 to i41
        %11 = arith.addi %9, %10 : i41
        allo.stream_put(%arg5, [], %11) : !allo.stream<i32, 2> contains i41
        %12 = affine.load %alloc[] {from = "a"} : memref<i8>
        allo.stream_put(%arg4, [], %12) : !allo.stream<i8, 2> contains i8
      } {loop_name = "t", op_name = "S_t_0"}
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @tiled_vpu_r0(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<12xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<12xi32>)
    affine.for %arg6 = 0 to 12 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<12xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg6 = 0 to 6 {
      %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_1 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_1 : i32) outs(%alloc_0 : memref<4xi32>)
      affine.for %arg7 = 0 to 12 {
        %0 = affine.load %alloc[%arg7] {from = "prog"} : memref<12xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_2[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_3 = arith.constant 24 : i32
        %2 = arith.shrsi %1, %c24_i32_3 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_4 = arith.constant 255 : i32
        %3 = arith.andi %2, %c255_i32_4 : i32
        %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_5[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_6 = arith.constant 20 : i32
        %5 = arith.shrsi %4, %c20_i32_6 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_7 = arith.constant 15 : i32
        %6 = arith.andi %5, %c15_i32_7 : i32
        %alloc_8 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_8[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_9 = arith.constant 16 : i32
        %8 = arith.shrsi %7, %c16_i32_9 : i32
        %c15_i32_10 = arith.constant 15 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %9 = arith.andi %8, %c15_i32_11 : i32
        %alloc_12 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_12[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_13 = arith.constant 65535 : i32
        %11 = arith.andi %10, %c65535_i32_13 : i32
        %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_14[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c9_i32 = arith.constant 9 : i32
        %c9_i32_15 = arith.constant 9 : i32
        %13 = arith.cmpi eq, %12, %c9_i32_15 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_16 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_16[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_16[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          %23 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %24 = arith.index_cast %23 : i32 to index
          memref.store %22, %alloc_0[%24] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_16 = arith.constant 2 : i32
          %15 = arith.cmpi eq, %14, %c2_i32_16 : i32
          scf.if %15 {
            %16 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %17 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
            %18 = arith.index_cast %17 : i32 to index
            memref.store %16, %alloc_0[%18] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_17 = arith.constant 3 : i32
            %17 = arith.cmpi eq, %16, %c3_i32_17 : i32
            scf.if %17 {
              %18 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              %19 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
              %20 = arith.index_cast %19 : i32 to index
              memref.store %18, %alloc_0[%20] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_18 = arith.constant 4 : i32
              %19 = arith.cmpi eq, %18, %c4_i32_18 : i32
              scf.if %19 {
                %20 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %21 = arith.index_cast %20 : i32 to index
                %22 = memref.load %alloc_0[%21] {from = "reg"} : memref<4xi32>
                %23 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                %24 = arith.index_cast %23 : i32 to index
                %25 = memref.load %alloc_0[%24] {from = "reg"} : memref<4xi32>
                %26 = arith.extsi %22 : i32 to i33
                %27 = arith.extsi %25 : i32 to i33
                %28 = arith.addi %26, %27 : i33
                %29 = arith.trunci %28 : i33 to i32
                %30 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %31 = arith.index_cast %30 : i32 to index
                memref.store %29, %alloc_0[%31] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_19 = arith.constant 5 : i32
                %21 = arith.cmpi eq, %20, %c5_i32_19 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i64
                  %29 = arith.extsi %27 : i32 to i64
                  %30 = arith.muli %28, %29 : i64
                  %31 = arith.trunci %30 : i64 to i32
                  %32 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %33 = arith.index_cast %32 : i32 to index
                  memref.store %31, %alloc_0[%33] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_20 = arith.constant 6 : i32
                  %23 = arith.cmpi eq, %22, %c6_i32_20 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.cmpi sgt, %26, %29 : i32
                    scf.if %30 {
                      %31 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      %33 = memref.load %alloc_0[%32] {from = "reg"} : memref<4xi32>
                      %34 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %35 = arith.index_cast %34 : i32 to index
                      memref.store %33, %alloc_0[%35] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %24 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_21 = arith.constant 7 : i32
                    %25 = arith.cmpi eq, %24, %c7_i32_21 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %30 = arith.shrsi %28, %29 : i32
                      %31 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      memref.store %30, %alloc_0[%32] {to = "reg"} : memref<4xi32>
                    } else {
                      %26 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_22 = arith.constant 8 : i32
                      %27 = arith.cmpi eq, %26, %c8_i32_22 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg4, [], %30) : !allo.stream<i32, 6> contains i32
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
  func.func @tiled_vpu_r1(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 2>, %arg3: !allo.stream<i32, 6>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<12xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<12xi32>)
    affine.for %arg5 = 0 to 12 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg5] {to = "prog"} : memref<12xi32>
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg5 = 0 to 6 {
      %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_1 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_1 : i32) outs(%alloc_0 : memref<4xi32>)
      affine.for %arg6 = 0 to 12 {
        %0 = affine.load %alloc[%arg6] {from = "prog"} : memref<12xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_2[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_3 = arith.constant 24 : i32
        %2 = arith.shrsi %1, %c24_i32_3 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_4 = arith.constant 255 : i32
        %3 = arith.andi %2, %c255_i32_4 : i32
        %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_5[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_6 = arith.constant 20 : i32
        %5 = arith.shrsi %4, %c20_i32_6 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_7 = arith.constant 15 : i32
        %6 = arith.andi %5, %c15_i32_7 : i32
        %alloc_8 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_8[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_9 = arith.constant 16 : i32
        %8 = arith.shrsi %7, %c16_i32_9 : i32
        %c15_i32_10 = arith.constant 15 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %9 = arith.andi %8, %c15_i32_11 : i32
        %alloc_12 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_12[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_13 = arith.constant 65535 : i32
        %11 = arith.andi %10, %c65535_i32_13 : i32
        %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_14[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c9_i32 = arith.constant 9 : i32
        %c9_i32_15 = arith.constant 9 : i32
        %13 = arith.cmpi eq, %12, %c9_i32_15 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
          %alloc_16 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_16[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_16[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          %23 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %24 = arith.index_cast %23 : i32 to index
          memref.store %22, %alloc_0[%24] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_16 = arith.constant 2 : i32
          %15 = arith.cmpi eq, %14, %c2_i32_16 : i32
          scf.if %15 {
            %16 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %17 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
            %18 = arith.index_cast %17 : i32 to index
            memref.store %16, %alloc_0[%18] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_17 = arith.constant 3 : i32
            %17 = arith.cmpi eq, %16, %c3_i32_17 : i32
            scf.if %17 {
              %18 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              %19 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
              %20 = arith.index_cast %19 : i32 to index
              memref.store %18, %alloc_0[%20] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_18 = arith.constant 4 : i32
              %19 = arith.cmpi eq, %18, %c4_i32_18 : i32
              scf.if %19 {
                %20 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %21 = arith.index_cast %20 : i32 to index
                %22 = memref.load %alloc_0[%21] {from = "reg"} : memref<4xi32>
                %23 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                %24 = arith.index_cast %23 : i32 to index
                %25 = memref.load %alloc_0[%24] {from = "reg"} : memref<4xi32>
                %26 = arith.extsi %22 : i32 to i33
                %27 = arith.extsi %25 : i32 to i33
                %28 = arith.addi %26, %27 : i33
                %29 = arith.trunci %28 : i33 to i32
                %30 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %31 = arith.index_cast %30 : i32 to index
                memref.store %29, %alloc_0[%31] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_19 = arith.constant 5 : i32
                %21 = arith.cmpi eq, %20, %c5_i32_19 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i64
                  %29 = arith.extsi %27 : i32 to i64
                  %30 = arith.muli %28, %29 : i64
                  %31 = arith.trunci %30 : i64 to i32
                  %32 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %33 = arith.index_cast %32 : i32 to index
                  memref.store %31, %alloc_0[%33] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_20 = arith.constant 6 : i32
                  %23 = arith.cmpi eq, %22, %c6_i32_20 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.cmpi sgt, %26, %29 : i32
                    scf.if %30 {
                      %31 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      %33 = memref.load %alloc_0[%32] {from = "reg"} : memref<4xi32>
                      %34 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %35 = arith.index_cast %34 : i32 to index
                      memref.store %33, %alloc_0[%35] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %24 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_21 = arith.constant 7 : i32
                    %25 = arith.cmpi eq, %24, %c7_i32_21 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %30 = arith.shrsi %28, %29 : i32
                      %31 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      memref.store %30, %alloc_0[%32] {to = "reg"} : memref<4xi32>
                    } else {
                      %26 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_22 = arith.constant 8 : i32
                      %27 = arith.cmpi eq, %26, %c8_i32_22 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg3, [], %30) : !allo.stream<i32, 6> contains i32
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
  func.func @tiled_vpu_r2(%arg0: memref<4xi32>, %arg1: index, %arg2: !allo.stream<i32, 12>, %arg3: !allo.stream<i32, 2>, %arg4: !allo.stream<i32, 6>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    %alloc = memref.alloc() {name = "prog"} : memref<12xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<12xi32>)
    affine.for %arg6 = 0 to 12 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 12> -> i32
      %alloc_0 = memref.alloc() {name = "word"} : memref<i32>
      affine.store %0, %alloc_0[] {to = "word"} : memref<i32>
      %1 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      affine.store %1, %alloc[%arg6] {to = "prog"} : memref<12xi32>
      %2 = affine.load %alloc_0[] {from = "word"} : memref<i32>
      allo.stream_put(%arg3, [], %2) : !allo.stream<i32, 2> contains i32
    } {loop_name = "pc", op_name = "S_pc_0"}
    affine.for %arg6 = 0 to 6 {
      %alloc_0 = memref.alloc() {name = "reg"} : memref<4xi32>
      %c0_i32_1 = arith.constant 0 : i32
      linalg.fill ins(%c0_i32_1 : i32) outs(%alloc_0 : memref<4xi32>)
      affine.for %arg7 = 0 to 12 {
        %0 = affine.load %alloc[%arg7] {from = "prog"} : memref<12xi32>
        %alloc_2 = memref.alloc() {name = "word2"} : memref<i32>
        affine.store %0, %alloc_2[] {to = "word2"} : memref<i32>
        %1 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c24_i32 = arith.constant 24 : i32
        %c24_i32_3 = arith.constant 24 : i32
        %2 = arith.shrsi %1, %c24_i32_3 : i32
        %c255_i32 = arith.constant 255 : i32
        %c255_i32_4 = arith.constant 255 : i32
        %3 = arith.andi %2, %c255_i32_4 : i32
        %alloc_5 = memref.alloc() {name = "opcode"} : memref<i32>
        affine.store %3, %alloc_5[] {to = "opcode"} : memref<i32>
        %4 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c20_i32 = arith.constant 20 : i32
        %c20_i32_6 = arith.constant 20 : i32
        %5 = arith.shrsi %4, %c20_i32_6 : i32
        %c15_i32 = arith.constant 15 : i32
        %c15_i32_7 = arith.constant 15 : i32
        %6 = arith.andi %5, %c15_i32_7 : i32
        %alloc_8 = memref.alloc() {name = "dst"} : memref<i32>
        affine.store %6, %alloc_8[] {to = "dst"} : memref<i32>
        %7 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c16_i32 = arith.constant 16 : i32
        %c16_i32_9 = arith.constant 16 : i32
        %8 = arith.shrsi %7, %c16_i32_9 : i32
        %c15_i32_10 = arith.constant 15 : i32
        %c15_i32_11 = arith.constant 15 : i32
        %9 = arith.andi %8, %c15_i32_11 : i32
        %alloc_12 = memref.alloc() {name = "src"} : memref<i32>
        affine.store %9, %alloc_12[] {to = "src"} : memref<i32>
        %10 = affine.load %alloc_2[] {from = "word2"} : memref<i32>
        %c65535_i32 = arith.constant 65535 : i32
        %c65535_i32_13 = arith.constant 65535 : i32
        %11 = arith.andi %10, %c65535_i32_13 : i32
        %alloc_14 = memref.alloc() {name = "imm"} : memref<i32>
        affine.store %11, %alloc_14[] {to = "imm"} : memref<i32>
        %12 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
        %c9_i32 = arith.constant 9 : i32
        %c9_i32_15 = arith.constant 9 : i32
        %13 = arith.cmpi eq, %12, %c9_i32_15 : i32
        scf.if %13 {
          %14 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
          %alloc_16 = memref.alloc() {name = "zz"} : memref<i32>
          affine.store %14, %alloc_16[] {to = "zz"} : memref<i32>
          %15 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %16 = arith.index_cast %15 : i32 to index
          %17 = memref.load %alloc_0[%16] {from = "reg"} : memref<4xi32>
          %18 = affine.load %alloc_16[] {from = "zz"} : memref<i32>
          %19 = arith.extsi %17 : i32 to i33
          %20 = arith.extsi %18 : i32 to i33
          %21 = arith.addi %19, %20 : i33
          %22 = arith.trunci %21 : i33 to i32
          %23 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
          %24 = arith.index_cast %23 : i32 to index
          memref.store %22, %alloc_0[%24] {to = "reg"} : memref<4xi32>
        } else {
          %14 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
          %c2_i32 = arith.constant 2 : i32
          %c2_i32_16 = arith.constant 2 : i32
          %15 = arith.cmpi eq, %14, %c2_i32_16 : i32
          scf.if %15 {
            %16 = affine.load %arg0[%arg1] {from = "local_Bias"} : memref<4xi32>
            %17 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
            %18 = arith.index_cast %17 : i32 to index
            memref.store %16, %alloc_0[%18] {to = "reg"} : memref<4xi32>
          } else {
            %16 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
            %c3_i32 = arith.constant 3 : i32
            %c3_i32_17 = arith.constant 3 : i32
            %17 = arith.cmpi eq, %16, %c3_i32_17 : i32
            scf.if %17 {
              %18 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
              %19 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
              %20 = arith.index_cast %19 : i32 to index
              memref.store %18, %alloc_0[%20] {to = "reg"} : memref<4xi32>
            } else {
              %18 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
              %c4_i32 = arith.constant 4 : i32
              %c4_i32_18 = arith.constant 4 : i32
              %19 = arith.cmpi eq, %18, %c4_i32_18 : i32
              scf.if %19 {
                %20 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %21 = arith.index_cast %20 : i32 to index
                %22 = memref.load %alloc_0[%21] {from = "reg"} : memref<4xi32>
                %23 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                %24 = arith.index_cast %23 : i32 to index
                %25 = memref.load %alloc_0[%24] {from = "reg"} : memref<4xi32>
                %26 = arith.extsi %22 : i32 to i33
                %27 = arith.extsi %25 : i32 to i33
                %28 = arith.addi %26, %27 : i33
                %29 = arith.trunci %28 : i33 to i32
                %30 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                %31 = arith.index_cast %30 : i32 to index
                memref.store %29, %alloc_0[%31] {to = "reg"} : memref<4xi32>
              } else {
                %20 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                %c5_i32 = arith.constant 5 : i32
                %c5_i32_19 = arith.constant 5 : i32
                %21 = arith.cmpi eq, %20, %c5_i32_19 : i32
                scf.if %21 {
                  %22 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %23 = arith.index_cast %22 : i32 to index
                  %24 = memref.load %alloc_0[%23] {from = "reg"} : memref<4xi32>
                  %25 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                  %26 = arith.index_cast %25 : i32 to index
                  %27 = memref.load %alloc_0[%26] {from = "reg"} : memref<4xi32>
                  %28 = arith.extsi %24 : i32 to i64
                  %29 = arith.extsi %27 : i32 to i64
                  %30 = arith.muli %28, %29 : i64
                  %31 = arith.trunci %30 : i64 to i32
                  %32 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                  %33 = arith.index_cast %32 : i32 to index
                  memref.store %31, %alloc_0[%33] {to = "reg"} : memref<4xi32>
                } else {
                  %22 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                  %c6_i32 = arith.constant 6 : i32
                  %c6_i32_20 = arith.constant 6 : i32
                  %23 = arith.cmpi eq, %22, %c6_i32_20 : i32
                  scf.if %23 {
                    %24 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloc_0[%25] {from = "reg"} : memref<4xi32>
                    %27 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                    %28 = arith.index_cast %27 : i32 to index
                    %29 = memref.load %alloc_0[%28] {from = "reg"} : memref<4xi32>
                    %30 = arith.cmpi sgt, %26, %29 : i32
                    scf.if %30 {
                      %31 = affine.load %alloc_12[] {from = "src"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      %33 = memref.load %alloc_0[%32] {from = "reg"} : memref<4xi32>
                      %34 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %35 = arith.index_cast %34 : i32 to index
                      memref.store %33, %alloc_0[%35] {to = "reg"} : memref<4xi32>
                    }
                  } else {
                    %24 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                    %c7_i32 = arith.constant 7 : i32
                    %c7_i32_21 = arith.constant 7 : i32
                    %25 = arith.cmpi eq, %24, %c7_i32_21 : i32
                    scf.if %25 {
                      %26 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %27 = arith.index_cast %26 : i32 to index
                      %28 = memref.load %alloc_0[%27] {from = "reg"} : memref<4xi32>
                      %29 = affine.load %alloc_14[] {from = "imm"} : memref<i32>
                      %30 = arith.shrsi %28, %29 : i32
                      %31 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                      %32 = arith.index_cast %31 : i32 to index
                      memref.store %30, %alloc_0[%32] {to = "reg"} : memref<4xi32>
                    } else {
                      %26 = affine.load %alloc_5[] {from = "opcode"} : memref<i32>
                      %c8_i32 = arith.constant 8 : i32
                      %c8_i32_22 = arith.constant 8 : i32
                      %27 = arith.cmpi eq, %26, %c8_i32_22 : i32
                      scf.if %27 {
                        %28 = affine.load %alloc_8[] {from = "dst"} : memref<i32>
                        %29 = arith.index_cast %28 : i32 to index
                        %30 = memref.load %alloc_0[%29] {from = "reg"} : memref<4xi32>
                        allo.stream_put(%arg4, [], %30) : !allo.stream<i32, 6> contains i32
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
  func.func @tiled_vpu_y_out_drain(%arg0: memref<6x4xi32>, %arg1: index, %arg2: !allo.stream<i32, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 6> -> i32
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<6x4xi32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<12x4xi8>, %arg1: memref<12xi32>, %arg2: memref<4x4x2xi8>, %arg3: memref<4xi32>, %arg4: memref<6x4xi32>) attributes {dataflow, itypes = "sssss", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [4], families = [#spmw.family<name = "tiled_mac_a_in_bind", type = i8, block = [], depth = 12, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "tiled_mac_a_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @tiled_mac_a_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<12x4xi8>
    spmw.map(%arg1) topology = <grid = [1], families = [#spmw.family<name = "tiled_vpu_op_in_bind", type = i32, block = [], depth = 12, shape = [1]>], ports = [#spmw.port_map<port = "chan", family = "tiled_vpu_op_in_bind", kind = "table", slots = dense<0> : tensor<1xi32>>]> roles = [#spmw.role<unit = @tiled_vpu_op_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<1xi32> : memref<12xi32>
    spmw.map(%arg2) topology = <grid = [4, 4], families = [#spmw.family<name = "tiled_mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "tiled_mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "tiled_mac_a_in_bind", type = i8, block = [], depth = 12, shape = [4]>, #spmw.family<name = "tiled_vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "a_in", family = "tiled_mac_a_in_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "tiled_vpu_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "tiled_vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "tiled_mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "tiled_mac_a_out_a_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "p_in", family = "tiled_mac_p_out_p_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "p_out", family = "tiled_mac_p_out_p_in", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @tiled_mac_r0, missing = [], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r1, missing = ["p_out"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r2, missing = ["p_in"], ports = ["a_in", "a_out", "p_out"]>, #spmw.role<unit = @tiled_mac_r3, missing = ["a_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r4, missing = ["a_in"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r5, missing = ["a_out", "p_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r6, missing = ["a_out", "p_in"], ports = ["a_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r7, missing = ["a_in", "p_out"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @tiled_mac_r8, missing = ["a_in", "p_in"], ports = ["a_in", "a_out", "p_out"]>] classes = dense<[8, 2, 2, 6, 4, 0, 0, 3, 4, 0, 0, 3, 7, 1, 1, 5]> : tensor<16xi32> : memref<4x4x2xi8>
    spmw.map(%arg3) topology = <grid = [4], families = [#spmw.family<name = "tiled_vpu_op_out_op_in", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "tiled_vpu_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "tiled_vpu_op_in_bind", type = i32, block = [], depth = 12, shape = [1]>, #spmw.family<name = "tiled_vpu_y_out_bind", type = i32, block = [], depth = 6, shape = [4]>], ports = [#spmw.port_map<port = "p_out", family = "tiled_vpu_z_in_bind", kind = "table", slots = dense<-1> : tensor<4xi32>>, #spmw.port_map<port = "z_in", family = "tiled_vpu_z_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "tiled_vpu_op_in_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "y_out", family = "tiled_vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "op_in", family = "tiled_vpu_op_out_op_in", kind = "affine", offset = [0]>, #spmw.port_map<port = "op_out", family = "tiled_vpu_op_out_op_in", kind = "affine", offset = [1]>]> roles = [#spmw.role<unit = @tiled_vpu_r0, missing = ["y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>, #spmw.role<unit = @tiled_vpu_r1, missing = ["op_out", "y_out", "z_in"], ports = ["op_in", "y_out", "z_in"]>, #spmw.role<unit = @tiled_vpu_r2, missing = ["op_in", "y_out", "z_in"], ports = ["op_in", "op_out", "y_out", "z_in"]>] classes = dense<[2, 0, 0, 1]> : tensor<4xi32> : memref<4xi32>
    spmw.map(%arg4) topology = <grid = [4], families = [#spmw.family<name = "tiled_vpu_y_out_bind", type = i32, block = [], depth = 6, shape = [4]>], ports = [#spmw.port_map<port = "chan", family = "tiled_vpu_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @tiled_vpu_y_out_drain, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<4xi32> : memref<6x4xi32>
    return
  }
}
