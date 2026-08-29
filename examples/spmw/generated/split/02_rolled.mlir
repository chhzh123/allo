module {
  func.func @pe_r0(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i8, 2>, %a8: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r1(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r2(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r3(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r4(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>, %a7: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r5(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r6(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r7(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>) {
    return
  }
  func.func @pe_r8(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i8, 2>, %a6: !allo.stream<i8, 2>) {
    return
  }
  func.func @drain_r0(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i32, 2>, %a6: !allo.stream<i32, 2>) {
    return
  }
  func.func @drain_r1(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i32, 2>) {
    return
  }
  func.func @drain_r2(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: index, %a5: !allo.stream<i32, 2>) {
    return
  }
  func.func @feed_r0(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>, %a5: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @feed_r1(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @feed_r2(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @feed_3_r0(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>, %a5: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @feed_3_r1(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @feed_3_r2(%a0: memref<4x4xi8>, %a1: memref<4x4xi8>, %a2: memref<4x4xi32>, %a3: index, %a4: !allo.stream<memref<4xi8>, 2>) {
    return
  }
  func.func @top(%t0: memref<4x4xi8>, %t1: memref<4x4xi8>, %t2: memref<4x4xi32>) attributes {dataflow} {
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4, 4], families = [#spmw.family<name = "pe_east_west", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "pe_south_north", type = i8, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 2, shape = [4]>, #spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 2, shape = [4]>, #spmw.family<name = "drain_mine_bind", type = i32, block = [], depth = 2, shape = [16]>], ports = [#spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "lane", family = "pe_west_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "lane", family = "pe_north_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]> : tensor<16xi32>>, #spmw.port_map<port = "c", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "mine", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>]> roles = [#spmw.role<unit = @pe_r0, missing = ["c"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r1, missing = ["c", "west"], ports = ["east", "north", "south"]>, #spmw.role<unit = @pe_r2, missing = ["c", "south"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r3, missing = ["c", "north"], ports = ["east", "south", "west"]>, #spmw.role<unit = @pe_r4, missing = ["c", "east"], ports = ["north", "south", "west"]>, #spmw.role<unit = @pe_r5, missing = ["c", "south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_r6, missing = ["c", "north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_r7, missing = ["c", "east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_r8, missing = ["c", "east", "north"], ports = ["south", "west"]>] classes = dense<[6, 3, 3, 8, 1, 0, 0, 4, 1, 0, 0, 4, 5, 2, 2, 7]> : tensor<16xi32> : memref<4x4xi8>, memref<4x4xi8>, memref<4x4xi32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4, 4], families = [#spmw.family<name = "drain_down_up", type = i32, block = [], depth = 2, shape = [4, 4]>, #spmw.family<name = "drain_mine_bind", type = i32, block = [], depth = 2, shape = [16]>, #spmw.family<name = "drain_down_bind", type = i32, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "down", family = "drain_down_up", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "up", family = "drain_down_up", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "c", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "mine", family = "drain_mine_bind", kind = "table", slots = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>>, #spmw.port_map<port = "down", family = "drain_down_bind", kind = "table", slots = dense<[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3]> : tensor<16xi32>>]> roles = [#spmw.role<unit = @drain_r0, missing = ["mine"], ports = ["down", "up"]>, #spmw.role<unit = @drain_r1, missing = ["mine", "up"], ports = ["down"]>, #spmw.role<unit = @drain_r2, missing = ["down", "mine"], ports = ["up"]>] classes = dense<[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]> : tensor<16xi32> : memref<4x4xi8>, memref<4x4xi8>, memref<4x4xi32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4], families = [#spmw.family<name = "feed_down_up", type = memref<4xi8>, block = [4], depth = 2, shape = [4]>, #spmw.family<name = "feed_up_bind", type = memref<4xi8>, block = [4], depth = 2, shape = [1]>, #spmw.family<name = "pe_west_bind", type = i8, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "down", family = "feed_down_up", kind = "affine", offset = [1]>, #spmw.port_map<port = "up", family = "feed_down_up", kind = "affine", offset = [0]>, #spmw.port_map<port = "up", family = "feed_up_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "lane", family = "pe_west_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[-1, -1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @feed_r0, missing = ["lane"], ports = ["down", "up"]>, #spmw.role<unit = @feed_r1, missing = ["lane", "up"], ports = ["down"]>, #spmw.role<unit = @feed_r2, missing = ["down", "lane"], ports = ["up"]>] classes = dense<[1, 0, 0, 2]> : tensor<4xi32> : memref<4x4xi8>, memref<4x4xi8>, memref<4x4xi32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [4], families = [#spmw.family<name = "feed_3_down_up", type = memref<4xi8>, block = [4], depth = 2, shape = [4]>, #spmw.family<name = "feed_3_up_bind", type = memref<4xi8>, block = [4], depth = 2, shape = [1]>, #spmw.family<name = "pe_north_bind", type = i8, block = [], depth = 2, shape = [4]>], ports = [#spmw.port_map<port = "down", family = "feed_3_down_up", kind = "affine", offset = [1]>, #spmw.port_map<port = "up", family = "feed_3_down_up", kind = "affine", offset = [0]>, #spmw.port_map<port = "up", family = "feed_3_up_bind", kind = "table", slots = dense<[0, -1, -1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "lane", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, 3]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[-1, -1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @feed_3_r0, missing = ["lane"], ports = ["down", "up"]>, #spmw.role<unit = @feed_3_r1, missing = ["lane", "up"], ports = ["down"]>, #spmw.role<unit = @feed_3_r2, missing = ["down", "lane"], ports = ["up"]>] classes = dense<[1, 0, 0, 2]> : tensor<4xi32> : memref<4x4xi8>, memref<4x4xi8>, memref<4x4xi32>
    return
  }
}
