module {
  func.func @pe_west_load(%arg0: memref<3x3xf32>, %arg1: index, %arg2: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg1, %arg3] {from = "local_A"} : memref<3x3xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 3> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_north_load(%arg0: memref<3x3xf32>, %arg1: index, %arg2: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_B"} : memref<3x3xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 3> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_r0(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r1(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
      %10 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %10) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r2(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
      %10 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %10) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r3(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r4(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 3>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 3> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
      %10 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %10) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r5(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 3>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 3> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %9) : !allo.stream<f32, 2> contains f32
      %10 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %10) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r6(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg4, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r7(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 3 {
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 2> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @pe_r8(%arg0: memref<3x3xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 3>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 3> -> f32
      %alloc_2 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %3, %alloc_2[] {to = "b"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "a"} : memref<f32>
      %5 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      %6 = arith.mulf %4, %5 : f32
      %7 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %alloc[] {to = "acc"} : memref<f32>
      %9 = affine.load %alloc_2[] {from = "b"} : memref<f32>
      allo.stream_put(%arg4, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0"}
    %1 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32>
    return
  }
  func.func @top(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) attributes {dataflow, itypes = "___", otypes = ""} {
    spmw.map(%arg0) topology = <grid = [3], families = [#spmw.family<name = "pe_west_bind", type = f32, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "chan", family = "pe_west_bind", kind = "table", slots = dense<[0, 1, 2]> : tensor<3xi32>>]> roles = [#spmw.role<unit = @pe_west_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<3xi32> : memref<3x3xf32>
    spmw.map(%arg1) topology = <grid = [3], families = [#spmw.family<name = "pe_north_bind", type = f32, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "chan", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2]> : tensor<3xi32>>]> roles = [#spmw.role<unit = @pe_north_load, missing = [], ports = ["chan"]>] classes = dense<0> : tensor<3xi32> : memref<3x3xf32>
    spmw.map(%arg2) topology = <grid = [3, 3], families = [#spmw.family<name = "pe_east_west", type = f32, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_south_north", type = f32, block = [], depth = 2, shape = [3, 3]>, #spmw.family<name = "pe_west_bind", type = f32, block = [], depth = 3, shape = [3]>, #spmw.family<name = "pe_north_bind", type = f32, block = [], depth = 3, shape = [3]>], ports = [#spmw.port_map<port = "west", family = "pe_west_bind", kind = "table", slots = dense<[0, -1, -1, 1, -1, -1, 2, -1, -1]> : tensor<9xi32>>, #spmw.port_map<port = "north", family = "pe_north_bind", kind = "table", slots = dense<[0, 1, 2, -1, -1, -1, -1, -1, -1]> : tensor<9xi32>>, #spmw.port_map<port = "east", family = "pe_east_west", kind = "affine", offset = [0, 1]>, #spmw.port_map<port = "west", family = "pe_east_west", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "north", family = "pe_south_north", kind = "affine", offset = [0, 0]>, #spmw.port_map<port = "south", family = "pe_south_north", kind = "affine", offset = [1, 0]>]> roles = [#spmw.role<unit = @pe_r0, missing = ["south", "west"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r1, missing = ["west"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r2, missing = [], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r3, missing = ["south"], ports = ["east", "north", "west"]>, #spmw.role<unit = @pe_r4, missing = ["north", "west"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r5, missing = ["north"], ports = ["east", "north", "south", "west"]>, #spmw.role<unit = @pe_r6, missing = ["east"], ports = ["north", "south", "west"]>, #spmw.role<unit = @pe_r7, missing = ["east", "south"], ports = ["north", "west"]>, #spmw.role<unit = @pe_r8, missing = ["east", "north"], ports = ["north", "south", "west"]>] classes = dense<[4, 5, 8, 1, 2, 6, 0, 3, 7]> : tensor<9xi32> : memref<3x3xf32>
    return
  }
}
