module {
  func.func @mac_a_in_load(%arg0: memref<6x8xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s___", otypes = ""} {
    affine.for %arg4 = 0 to 6 {
      %0 = affine.load %arg0[%arg4, %arg2 * 4 + %arg1] {from = "local_Pr"} : memref<6x8xi8>
      allo.stream_put(%arg3, [], %0) : !allo.stream<i8, 6> contains i8
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @mac_r0(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
  func.func @mac_r1(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
  func.func @mac_r2(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
    affine.for %arg6 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 2> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg4, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
  func.func @mac_r3(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 2>, %arg4: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s____", otypes = ""} {
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
      %3 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
  func.func @mac_r4(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>, %arg6: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s______", otypes = ""} {
    affine.for %arg7 = 0 to 6 {
      %0 = allo.stream_get(%arg3, []) : !allo.stream<i8, 6> -> i8
      %alloc = memref.alloc() {name = "a"} : memref<i8>
      affine.store %0, %alloc[] {to = "a"} : memref<i8>
      %1 = allo.stream_get(%arg5, []) : !allo.stream<i32, 2> -> i32
      %alloc_0 = memref.alloc() {name = "p"} : memref<i32>
      affine.store %1, %alloc_0[] {to = "p"} : memref<i32>
      %2 = affine.load %alloc_0[] {from = "p"} : memref<i32>
      %3 = affine.load %alloc[] {from = "a"} : memref<i8>
      %4 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
  func.func @mac_r5(%arg0: memref<8x2xi8>, %arg1: index, %arg2: index, %arg3: !allo.stream<i8, 6>, %arg4: !allo.stream<i8, 2>, %arg5: !allo.stream<i32, 2>) attributes {df.kernel, itypes = "s_____", otypes = ""} {
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
      %3 = affine.load %arg0[(%arg2 floordiv 2) * 4 + %arg1, %arg2 mod 2] {from = "local_V"} : memref<8x2xi8>
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
      %c2_i32 = arith.constant 2 : i32
      %c2_i32_1 = arith.constant 2 : i32
      %4 = arith.shrsi %3, %c2_i32_1 : i32
      %5 = arith.trunci %4 : i32 to i8
      %alloc_2 = memref.alloc() {name = "y"} : memref<i8>
      affine.store %5, %alloc_2[] {to = "y"} : memref<i8>
      %6 = affine.load %alloc_2[] {from = "y"} : memref<i8>
      allo.stream_put(%arg1, [], %6) : !allo.stream<i8, 6> contains i8
    } {loop_name = "m", op_name = "S_m_0"}
    return
  }
  func.func @act_y_out_drain(%arg0: memref<6x2xi8>, %arg1: index, %arg2: !allo.stream<i8, 6>) attributes {df.kernel, itypes = "s__", otypes = ""} {
    affine.for %arg3 = 0 to 6 {
      %0 = allo.stream_get(%arg2, []) : !allo.stream<i8, 6> -> i8
      affine.store %0, %arg0[%arg3, %arg1] {to = "local_Y"} : memref<6x2xi8>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<6x8xi8>, %arg1: memref<8x2xi8>, %arg2: memref<6x2xi8>) attributes {dataflow, itypes = "sss", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [4, 2], families = [#spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 6, shape = [8]>], ports = [#spmw.port_map<port = "chan", family = "mac_a_in_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>>]> roles = [#spmw.role<unit = @mac_a_in_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<8xi32> : memref<6x8xi8>
    spmw.map(%arg1) topology = <grid = [4, 4], families = [#spmw.family<name = "mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [14]>, #spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 6, shape = [8]>, #spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "p_in", family = "mac_p_out_p_in", kind = "table", slots = dense<[-1, -1, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "mac_p_out_p_in", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "mac_a_in_bind", kind = "table", slots = dense<[0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<-1> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "mac_a_out_a_in", kind = "affine", offset = [0, 1]>]> roles = [#spmw.role<unit = @mac_r0, missing = ["a_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r1, missing = ["a_in"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r2, missing = ["a_out", "p_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r3, missing = ["a_out", "p_in"], ports = ["a_in", "p_out"]>, #spmw.role<unit = @mac_r4, missing = ["a_in", "p_out"], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r5, missing = ["a_in", "p_in"], ports = ["a_in", "a_out", "p_out"]>] classes = dense<[5, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 4, 2]> : tensor<16xi32> : memref<8x2xi8>
    spmw.map() topology = <grid = [2], families = [#spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "act_y_out_bind", type = i8, block = [], depth = 6, shape = [2]>], ports = [#spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<-1> : tensor<2xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<[0, 1]> : tensor<2xi32>>, #spmw.port_map<port = "y_out", family = "act_y_out_bind", kind = "table", slots = dense<[0, 1]> : tensor<2xi32>>]> roles = [#spmw.role<unit = @act_r0, missing = ["y_out", "z_in"], ports = ["y_out", "z_in"]>] classes = dense<0> : tensor<2xi32>
    spmw.map(%arg2) topology = <grid = [2], families = [#spmw.family<name = "act_y_out_bind", type = i8, block = [], depth = 6, shape = [2]>], ports = [#spmw.port_map<port = "chan", family = "act_y_out_bind", kind = "table", slots = dense<[0, 1]> : tensor<2xi32>>]> roles = [#spmw.role<unit = @act_y_out_drain, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<2xi32> : memref<6x2xi8>
    return
  }
}
