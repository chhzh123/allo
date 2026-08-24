module {
  func.func @mac_r0(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i32, 2>, %a8: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r1(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r2(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r3(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r4(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r5(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r6(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r7(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r8(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @act_r0(%a0: memref<6x4xi8>, %a1: memref<4x4xi8>, %a2: memref<6x4xi8>, %a3: index) {
    return
  }
  func.func @top(%t0: memref<6x4xi8>, %t1: memref<4x4xi8>, %t2: memref<6x4xi8>) attributes {dataflow} {
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4, 4], families = [#spmw.family<name = "mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 2, shape = [4]>, #spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "a_in", family = "mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "mac_a_out_a_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "p_in", family = "mac_p_out_p_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "p_out", family = "mac_p_out_p_in", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "a_in", family = "mac_a_in_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>]> roles = [#spmw.role<unit = @mac_r0, missing = [], ports = ["a_in", "a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r1, missing = ["p_out"], ports = ["a_in", "a_out", "p_in"]>, #spmw.role<unit = @mac_r2, missing = ["p_in"], ports = ["a_in", "a_out", "p_out"]>, #spmw.role<unit = @mac_r3, missing = ["a_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r4, missing = ["a_in"], ports = ["a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r5, missing = ["a_out", "p_out"], ports = ["a_in", "p_in"]>, #spmw.role<unit = @mac_r6, missing = ["a_out", "p_in"], ports = ["a_in", "p_out"]>, #spmw.role<unit = @mac_r7, missing = ["a_in", "p_out"], ports = ["a_out", "p_in"]>, #spmw.role<unit = @mac_r8, missing = ["a_in", "p_in"], ports = ["a_out", "p_out"]>] classes = dense<[8, 2, 2, 6, 4, 0, 0, 3, 4, 0, 0, 3, 7, 1, 1, 5]> : tensor<16xi32> : memref<6x4xi8>, memref<4x4xi8>, memref<6x4xi8>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4], families = [#spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [4]>, #spmw.family<name = "act_y_out_bind", type = i8, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "y_out", family = "act_y_out_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @act_r0, missing = ["y_out", "z_in"], ports = []>] classes = dense<[0, 0, 0, 0]> : tensor<4xi32> : memref<6x4xi8>, memref<4x4xi8>, memref<6x4xi8>
    return
  }
}
