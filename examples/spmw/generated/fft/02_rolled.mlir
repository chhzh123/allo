module {
  func.func @bfly_r0(%a0: memref<8x2xf32>, %a1: memref<8x2xf32>, %a2: index, %a3: index, %a4: !allo.stream<memref<2xf32>, 2>, %a5: !allo.stream<memref<2xf32>, 2>, %a6: !allo.stream<memref<2xf32>, 2>, %a7: !allo.stream<memref<2xf32>, 2>) {
    return
  }
  func.func @bfly_r1(%a0: memref<8x2xf32>, %a1: memref<8x2xf32>, %a2: index, %a3: index, %a4: !allo.stream<memref<2xf32>, 2>, %a5: !allo.stream<memref<2xf32>, 2>) {
    return
  }
  func.func @bfly_r2(%a0: memref<8x2xf32>, %a1: memref<8x2xf32>, %a2: index, %a3: index, %a4: !allo.stream<memref<2xf32>, 2>, %a5: !allo.stream<memref<2xf32>, 2>) {
    return
  }
  func.func @top(%t0: memref<8x2xf32>, %t1: memref<8x2xf32>) attributes {dataflow} {
    spmw.map (%t0, %t1) topology = #spmw.topology<grid = [3, 4], families = [#spmw.family<name = "bfly_key", type = memref<2xf32>, block = [2], depth = 2, shape = [16]>, #spmw.family<name = "bfly_up_in_bind", type = memref<2xf32>, block = [2], depth = 2, shape = [4]>, #spmw.family<name = "bfly_lo_in_bind", type = memref<2xf32>, block = [2], depth = 2, shape = [4]>, #spmw.family<name = "bfly_up_out_bind", type = memref<2xf32>, block = [2], depth = 2, shape = [4]>, #spmw.family<name = "bfly_lo_out_bind", type = memref<2xf32>, block = [2], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "lo_in", family = "bfly_key", kind = "table", slots = dense<[-1, -1, -1, -1, 2, 3, 6, 7, 12, 13, 14, 15]> : tensor<12xi32>>, #spmw.port_map<port = "lo_out", family = "bfly_key", kind = "table", slots = dense<[1, 3, 5, 7, 10, 11, 14, 15, -1, -1, -1, -1]> : tensor<12xi32>>, #spmw.port_map<port = "up_in", family = "bfly_key", kind = "table", slots = dense<[-1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 10, 11]> : tensor<12xi32>>, #spmw.port_map<port = "up_out", family = "bfly_key", kind = "table", slots = dense<[0, 2, 4, 6, 8, 9, 12, 13, -1, -1, -1, -1]> : tensor<12xi32>>, #spmw.port_map<port = "up_in", family = "bfly_up_in_bind", kind = "table", slots = dense<[0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<12xi32>>, #spmw.port_map<port = "lo_in", family = "bfly_lo_in_bind", kind = "table", slots = dense<[0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<12xi32>>, #spmw.port_map<port = "up_out", family = "bfly_up_out_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<12xi32>>, #spmw.port_map<port = "lo_out", family = "bfly_lo_out_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<12xi32>>]> roles = [#spmw.role<unit = @bfly_r0, missing = [], ports = ["lo_in", "lo_out", "up_in", "up_out"]>, #spmw.role<unit = @bfly_r1, missing = ["lo_out", "up_out"], ports = ["lo_in", "up_in"]>, #spmw.role<unit = @bfly_r2, missing = ["lo_in", "up_in"], ports = ["lo_out", "up_out"]>] classes = dense<[2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]> : tensor<12xi32> : memref<8x2xf32>, memref<8x2xf32>
    return
  }
}
