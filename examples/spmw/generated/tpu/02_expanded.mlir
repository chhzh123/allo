module {
  func.func @mac_a_in_load(%arg0: memref<6x4xi8>, %arg1: index, %arg2: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_A"} : memref<6x4xi8>
      allo.stream_put(%arg2, [], %0) : !allo.stream<i8, 6> contains i8
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
  func.func @act_r0(%arg0: index, %arg1: !allo.stream<i8, 6>, %arg2: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i32, 2> -> i32
      %alloc = memref.alloc() {name = "z"} : memref<i32>
      affine.store %0, %alloc[] {to = "z"} : memref<i32>
      %1 = affine.load %alloc[] {from = "z"} : memref<i32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_0 = arith.constant 0 : i32
      %2 = arith.cmpi slt, %1, %c0_i32_0 : i32
      scf.if %2 {
        %c0_i32_3 = arith.constant 0 : i32
        %c0_i32_4 = arith.constant 0 : i32
        affine.store %c0_i32_4, %alloc[] {to = "z"} : memref<i32>
      }
      %3 = affine.load %alloc[] {from = "z"} : memref<i32>
      %c4_i32 = arith.constant 4 : i32
      %c4_i32_1 = arith.constant 4 : i32
      %4 = arith.shrsi %3, %c4_i32_1 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_2 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_2[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_2[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 6> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @act_y_out_drain(%arg0: memref<6x4xi8>, %arg1: index, %arg2: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i8, 6> -> i8
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<6x4xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<6x4xi8>, %arg1: memref<4x4xi8>, %arg2: memref<6x4xi8>) attributes {dataflow, itypes = "sss", otypes = ""} {
    %0 = allo.stream_construct() {name = "act_y_out_bind_3"} : !allo.stream<i8, 6>
    %1 = allo.stream_construct() {name = "act_y_out_bind_2"} : !allo.stream<i8, 6>
    %2 = allo.stream_construct() {name = "act_y_out_bind_1"} : !allo.stream<i8, 6>
    %3 = allo.stream_construct() {name = "act_y_out_bind_0"} : !allo.stream<i8, 6>
    %4 = allo.stream_construct() {name = "act_z_in_bind_3"} : !allo.stream<i32, 2>
    %5 = allo.stream_construct() {name = "act_z_in_bind_2"} : !allo.stream<i32, 2>
    %6 = allo.stream_construct() {name = "mac_a_out_a_in_3_3"} : !allo.stream<i8, 2>
    %7 = allo.stream_construct() {name = "act_z_in_bind_1"} : !allo.stream<i32, 2>
    %8 = allo.stream_construct() {name = "mac_a_out_a_in_3_2"} : !allo.stream<i8, 2>
    %9 = allo.stream_construct() {name = "act_z_in_bind_0"} : !allo.stream<i32, 2>
    %10 = allo.stream_construct() {name = "mac_a_out_a_in_3_1"} : !allo.stream<i8, 2>
    %11 = allo.stream_construct() {name = "mac_p_out_p_in_3_3"} : !allo.stream<i32, 2>
    %12 = allo.stream_construct() {name = "mac_p_out_p_in_3_2"} : !allo.stream<i32, 2>
    %13 = allo.stream_construct() {name = "mac_a_out_a_in_2_3"} : !allo.stream<i8, 2>
    %14 = allo.stream_construct() {name = "mac_p_out_p_in_3_1"} : !allo.stream<i32, 2>
    %15 = allo.stream_construct() {name = "mac_a_out_a_in_2_2"} : !allo.stream<i8, 2>
    %16 = allo.stream_construct() {name = "mac_p_out_p_in_3_0"} : !allo.stream<i32, 2>
    %17 = allo.stream_construct() {name = "mac_a_out_a_in_2_1"} : !allo.stream<i8, 2>
    %18 = allo.stream_construct() {name = "mac_p_out_p_in_2_3"} : !allo.stream<i32, 2>
    %19 = allo.stream_construct() {name = "mac_p_out_p_in_2_2"} : !allo.stream<i32, 2>
    %20 = allo.stream_construct() {name = "mac_a_out_a_in_1_3"} : !allo.stream<i8, 2>
    %21 = allo.stream_construct() {name = "mac_p_out_p_in_2_1"} : !allo.stream<i32, 2>
    %22 = allo.stream_construct() {name = "mac_a_out_a_in_1_2"} : !allo.stream<i8, 2>
    %23 = allo.stream_construct() {name = "mac_p_out_p_in_2_0"} : !allo.stream<i32, 2>
    %24 = allo.stream_construct() {name = "mac_a_out_a_in_1_1"} : !allo.stream<i8, 2>
    %25 = allo.stream_construct() {name = "mac_p_out_p_in_1_3"} : !allo.stream<i32, 2>
    %26 = allo.stream_construct() {name = "mac_p_out_p_in_1_2"} : !allo.stream<i32, 2>
    %27 = allo.stream_construct() {name = "mac_a_out_a_in_0_3"} : !allo.stream<i8, 2>
    %28 = allo.stream_construct() {name = "mac_p_out_p_in_1_1"} : !allo.stream<i32, 2>
    %29 = allo.stream_construct() {name = "mac_a_out_a_in_0_2"} : !allo.stream<i8, 2>
    %30 = allo.stream_construct() {name = "mac_p_out_p_in_1_0"} : !allo.stream<i32, 2>
    %31 = allo.stream_construct() {name = "mac_a_out_a_in_0_1"} : !allo.stream<i8, 2>
    %32 = allo.stream_construct() {name = "mac_a_in_bind_3"} : !allo.stream<i8, 6>
    %c3 = arith.constant 3 : index
    %33 = allo.stream_construct() {name = "mac_a_in_bind_2"} : !allo.stream<i8, 6>
    %c2 = arith.constant 2 : index
    %34 = allo.stream_construct() {name = "mac_a_in_bind_1"} : !allo.stream<i8, 6>
    %c1 = arith.constant 1 : index
    %35 = allo.stream_construct() {name = "mac_a_in_bind_0"} : !allo.stream<i8, 6>
    %c0 = arith.constant 0 : index
    call @mac_a_in_load(%arg0, %c0, %35) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c1, %34) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c2, %33) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @mac_a_in_load(%arg0, %c3, %32) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @mac_r8(%arg1, %c0, %c0, %35, %31, %30) : (memref<4x4xi8>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg1, %c0, %c1, %31, %29, %28) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r2(%arg1, %c0, %c2, %29, %27, %26) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r6(%arg1, %c0, %c3, %27, %25) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg1, %c1, %c0, %34, %24, %30, %23) : (memref<4x4xi8>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg1, %c1, %c1, %24, %22, %28, %21) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg1, %c1, %c2, %22, %20, %26, %19) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg1, %c1, %c3, %20, %25, %18) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r4(%arg1, %c2, %c0, %33, %17, %23, %16) : (memref<4x4xi8>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg1, %c2, %c1, %17, %15, %21, %14) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r0(%arg1, %c2, %c2, %15, %13, %19, %12) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r3(%arg1, %c2, %c3, %13, %18, %11) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r7(%arg1, %c3, %c0, %32, %10, %16, %9) : (memref<4x4xi8>, index, index, !allo.stream<i8, 6>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg1, %c3, %c1, %10, %8, %14, %7) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r1(%arg1, %c3, %c2, %8, %6, %12, %5) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @mac_r5(%arg1, %c3, %c3, %6, %11, %4) : (memref<4x4xi8>, index, index, !allo.stream<i8, 2>, !allo.stream<i32, 2>, !allo.stream<i32, 2>) -> ()
    call @act_r0(%c0, %3, %9) : (index, !allo.stream<i8, 6>, !allo.stream<i32, 2>) -> ()
    call @act_r0(%c1, %2, %7) : (index, !allo.stream<i8, 6>, !allo.stream<i32, 2>) -> ()
    call @act_r0(%c2, %1, %5) : (index, !allo.stream<i8, 6>, !allo.stream<i32, 2>) -> ()
    call @act_r0(%c3, %0, %4) : (index, !allo.stream<i8, 6>, !allo.stream<i32, 2>) -> ()
    call @act_y_out_drain(%arg2, %c0, %3) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @act_y_out_drain(%arg2, %c1, %2) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @act_y_out_drain(%arg2, %c2, %1) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    call @act_y_out_drain(%arg2, %c3, %0) : (memref<6x4xi8>, index, !allo.stream<i8, 6>) -> ()
    return
  }
}
