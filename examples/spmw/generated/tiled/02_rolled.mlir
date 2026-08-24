module {
  func.func @pe_r0(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r1(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r2(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r3(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_1_r0(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_1_r1(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_1_r2(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_1_r3(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_2_r0(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_2_r1(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_2_r2(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_2_r3(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_3_r0(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_3_r1(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_3_r2(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_3_r3(%a0: memref<4x4xf32>, %a1: memref<4x4xf32>, %a2: memref<4x4xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @top(%t0: memref<4x4xf32>, %t1: memref<4x4xf32>, %t2: memref<4x4xf32>) attributes {dataflow} {
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [2, 2], families = [#spmw.family<name = "pe_east_west", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_south_north", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_west_bind", type = f32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "pe_north_bind", type = f32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, 1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @pe_r0, missing = ["south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_r1, missing = ["north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_r2, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_r3, missing = ["east", "north"], ports = ["south", "west"]>] classes = dense<[1, 3, 0, 2]> : tensor<4xi32> : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [2, 2], families = [#spmw.family<name = "pe_1_east_west", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_1_south_north", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_1_west_bind", type = f32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "pe_1_north_bind", type = f32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "east", family = "pe_1_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_1_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_1_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_1_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "west", family = "pe_1_west_bind", kind = "table", slots = dense<[0, -1, 1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_1_north_bind", kind = "table", slots = dense<[0, 1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @pe_1_r0, missing = ["south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_1_r1, missing = ["north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_1_r2, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_1_r3, missing = ["east", "north"], ports = ["south", "west"]>] classes = dense<[1, 3, 0, 2]> : tensor<4xi32> : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [2, 2], families = [#spmw.family<name = "pe_2_east_west", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_2_south_north", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_2_west_bind", type = f32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "pe_2_north_bind", type = f32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "east", family = "pe_2_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_2_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_2_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_2_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "west", family = "pe_2_west_bind", kind = "table", slots = dense<[0, -1, 1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_2_north_bind", kind = "table", slots = dense<[0, 1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @pe_2_r0, missing = ["south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_2_r1, missing = ["north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_2_r2, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_2_r3, missing = ["east", "north"], ports = ["south", "west"]>] classes = dense<[1, 3, 0, 2]> : tensor<4xi32> : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [2, 2], families = [#spmw.family<name = "pe_3_east_west", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_3_south_north", type = f32, block = [], depth = 2, shape = [2, 2]>, #spmw.family<name = "pe_3_west_bind", type = f32, block = [], depth = 2, shape = [2]>, #spmw.family<name = "pe_3_north_bind", type = f32, block = [], depth = 2, shape = [2]>], ports = [#spmw.port_map<port = "east", family = "pe_3_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_3_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_3_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_3_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "west", family = "pe_3_west_bind", kind = "table", slots = dense<[0, -1, 1, -1]> : tensor<4xi32>>, #spmw.port_map<port = "north", family = "pe_3_north_bind", kind = "table", slots = dense<[0, 1, -1, -1]> : tensor<4xi32>>]> roles = [#spmw.role<unit = @pe_3_r0, missing = ["south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_3_r1, missing = ["north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_3_r2, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_3_r3, missing = ["east", "north"], ports = ["south", "west"]>] classes = dense<[1, 3, 0, 2]> : tensor<4xi32> : memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>
    return
  }
}
