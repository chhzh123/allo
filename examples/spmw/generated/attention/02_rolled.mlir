module {
  func.func @mac_r0(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r1(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>, %a7: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r2(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r3(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r4(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @mac_r5(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @act_r0(%a0: memref<6x8xi8>, %a1: memref<8x2xi8>, %a2: memref<6x2xi8>, %a3: index) {
    return
  }
  func.func @top(%t0: memref<6x8xi8>, %t1: memref<8x2xi8>, %t2: memref<6x2xi8>) attributes {dataflow} {
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4, 4], families = [#spmw.family<name = "mac_a_out_a_in", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "mac_p_out_p_in", type = i32, block = [], depth = 2, shape = [14]>, #spmw.family<name = "mac_a_in_bind", type = i8, block = [], depth = 2, shape = [8]>, #spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "a_in", family = "mac_a_out_a_in", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "a_out", family = "mac_a_out_a_in", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "p_in", family = "mac_p_out_p_in", kind = "table", slots = dense<[-1, -1, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "mac_p_out_p_in", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "a_in", family = "mac_a_in_bind", kind = "table", slots = dense<[0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1]> : tensor<16xi32>>, #spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1]> : tensor<16xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>]> roles = [#spmw.role<unit = @mac_r0, missing = ["a_out"], ports = ["a_in", "p_in", "p_out"]>, #spmw.role<unit = @mac_r1, missing = ["a_in"], ports = ["a_out", "p_in", "p_out"]>, #spmw.role<unit = @mac_r2, missing = ["a_out", "p_out"], ports = ["a_in", "p_in"]>, #spmw.role<unit = @mac_r3, missing = ["a_out", "p_in"], ports = ["a_in", "p_out"]>, #spmw.role<unit = @mac_r4, missing = ["a_in", "p_out"], ports = ["a_out", "p_in"]>, #spmw.role<unit = @mac_r5, missing = ["a_in", "p_in"], ports = ["a_out", "p_out"]>] classes = dense<[5, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 4, 2]> : tensor<16xi32> : memref<6x8xi8>, memref<8x2xi8>, memref<6x2xi8>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [2], families = [#spmw.family<name = "act_z_in_bind", type = i32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "act_y_out_bind", type = i8, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "p_out", family = "act_z_in_bind", kind = "table", slots = dense<[-1, -1]> : tensor<2xi32>>, #spmw.port_map<port = "z_in", family = "act_z_in_bind", kind = "table", slots = dense<[0, 1]> : tensor<2xi32>>, #spmw.port_map<port = "y_out", family = "act_y_out_bind", kind = "table", slots = dense<[0, 1]> : tensor<2xi32>>]> roles = [#spmw.role<unit = @act_r0, missing = ["y_out", "z_in"], ports = []>] classes = dense<[0, 0]> : tensor<2xi32> : memref<6x8xi8>, memref<8x2xi8>, memref<6x2xi8>
    return
  }
}
