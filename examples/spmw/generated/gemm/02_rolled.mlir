module {
  func.func @pe_r0(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r1(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>, %a7: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r2(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>, %a7: !allo.stream<f32, 2>, %a8: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r3(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>, %a7: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r4(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r5(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>, %a7: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r6(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>, %a7: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r7(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @pe_r8(%a0: memref<3x3xf32>, %a1: memref<3x3xf32>, %a2: memref<3x3xf32>, %a3: index, %a4: index, %a5: !allo.stream<f32, 2>, %a6: !allo.stream<f32, 2>) {
    return
  }
  func.func @top(%t0: memref<3x3xf32>, %t1: memref<3x3xf32>, %t2: memref<3x3xf32>) attributes {dataflow} {
    spmw.map (%t0, %t1, %t2) topology = #spmw.topology<grid = [3, 3], families = [#spmw.family<name = "pe_east_west", type = f32, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_south_north", type = f32, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_west_bind", type = f32, block = [], depth = 2, shape = [3]>, #spmw.family<name = "pe_north_bind", type = f32, block = [], depth = 2, shape = [3]>], ports = [#spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>, #spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, -1, 1, -1, -1, 2, -1, -1]> : tensor<9xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, -1, -1, -1, -1, -1, -1]> : tensor<9xi32>>]> roles = [#spmw.role<unit = @pe_r0, missing = ["south", "west"], ports = ["east", "north"]>, #spmw.role<unit = @pe_r1, missing = ["west"], ports = ["east", "north", "south"]>, #spmw.role<unit = @pe_r2, missing = [], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r3, missing = ["south"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r4, missing = ["north", "west"], ports = ["east", "south"]>, #spmw.role<unit = @pe_r5, missing = ["north"], ports = ["east", "south", "west"]>, #spmw.role<unit = @pe_r6, missing = ["east"], ports = ["north", "south", "west"]>, #spmw.role<unit = @pe_r7, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_r8, missing = ["east", "north"], ports = ["south", "west"]>] classes = dense<[4, 5, 8, 1, 2, 6, 0, 3, 7]> : tensor<9xi32> : memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>
    return
  }
}
