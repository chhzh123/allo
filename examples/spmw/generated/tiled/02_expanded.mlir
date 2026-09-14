module {
  func.func @pe_west_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg1, %arg3] {from = "local_A"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_north_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_B"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_1_west_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg1, %arg3] {from = "local_A"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_1_north_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1 + 2] {from = "local_B"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_2_west_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg1 + 2, %arg3] {from = "local_A"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_2_north_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_B"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_3_west_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg1 + 2, %arg3] {from = "local_A"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_3_north_load(%arg0: memref<4x4xf32>, %arg1: index, %arg2: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.load %arg0[%arg3, %arg1 + 2] {from = "local_B"} : memref<4x4xf32>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 4> contains f32
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @pe_r0(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_r1(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 4>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 4 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 4> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_r2(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 4 {
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
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_r3(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 4>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_1_r0(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_1_r1(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 4>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 4 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 4> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_1_r2(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 4 {
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
    affine.store %1, %arg0[%arg1, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_1_r3(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 4>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_2_r0(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_2_r1(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 4>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 4 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 4> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_2_r2(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 4 {
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
    affine.store %1, %arg0[%arg1 + 2, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_2_r3(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 4>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_3_r0(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_3_r1(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 4>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 4>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 4 {
      %2 = allo.stream_get(%arg6, []) : !allo.stream<f32, 4> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg4, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_3_r2(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 4 {
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
    affine.store %1, %arg0[%arg1 + 2, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @pe_3_r3(%arg0: memref<4x4xf32>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 4>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %0 = arith.sitofp %c0_i32_0 : i32 to f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %0, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 4 {
      %2 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "a"} : memref<f32>
      %3 = allo.stream_get(%arg3, []) : !allo.stream<f32, 4> -> f32
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
    affine.store %1, %arg0[%arg1 + 2, %arg2 + 2] {to = "local_C"} : memref<4x4xf32>
    return
  }
  func.func @top(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) attributes {dataflow, itypes = "___", otypes = ""} {
    %0 = allo.stream_construct() {name = "pe_3_east_west_1_1"} : !allo.stream<f32, 2>
    %1 = allo.stream_construct() {name = "pe_3_south_north_1_1"} : !allo.stream<f32, 2>
    %2 = allo.stream_construct() {name = "pe_3_south_north_1_0"} : !allo.stream<f32, 2>
    %3 = allo.stream_construct() {name = "pe_3_east_west_0_1"} : !allo.stream<f32, 2>
    %4 = allo.stream_construct() {name = "pe_2_east_west_1_1"} : !allo.stream<f32, 2>
    %5 = allo.stream_construct() {name = "pe_2_south_north_1_1"} : !allo.stream<f32, 2>
    %6 = allo.stream_construct() {name = "pe_2_south_north_1_0"} : !allo.stream<f32, 2>
    %7 = allo.stream_construct() {name = "pe_2_east_west_0_1"} : !allo.stream<f32, 2>
    %8 = allo.stream_construct() {name = "pe_1_east_west_1_1"} : !allo.stream<f32, 2>
    %9 = allo.stream_construct() {name = "pe_1_south_north_1_1"} : !allo.stream<f32, 2>
    %10 = allo.stream_construct() {name = "pe_1_south_north_1_0"} : !allo.stream<f32, 2>
    %11 = allo.stream_construct() {name = "pe_1_east_west_0_1"} : !allo.stream<f32, 2>
    %12 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<f32, 2>
    %13 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<f32, 2>
    %14 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<f32, 2>
    %15 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<f32, 2>
    %16 = allo.stream_construct() {name = "pe_3_north_bind_1"} : !allo.stream<f32, 4>
    %17 = allo.stream_construct() {name = "pe_3_north_bind_0"} : !allo.stream<f32, 4>
    %18 = allo.stream_construct() {name = "pe_3_west_bind_1"} : !allo.stream<f32, 4>
    %19 = allo.stream_construct() {name = "pe_3_west_bind_0"} : !allo.stream<f32, 4>
    %20 = allo.stream_construct() {name = "pe_2_north_bind_1"} : !allo.stream<f32, 4>
    %21 = allo.stream_construct() {name = "pe_2_north_bind_0"} : !allo.stream<f32, 4>
    %22 = allo.stream_construct() {name = "pe_2_west_bind_1"} : !allo.stream<f32, 4>
    %23 = allo.stream_construct() {name = "pe_2_west_bind_0"} : !allo.stream<f32, 4>
    %24 = allo.stream_construct() {name = "pe_1_north_bind_1"} : !allo.stream<f32, 4>
    %25 = allo.stream_construct() {name = "pe_1_north_bind_0"} : !allo.stream<f32, 4>
    %26 = allo.stream_construct() {name = "pe_1_west_bind_1"} : !allo.stream<f32, 4>
    %27 = allo.stream_construct() {name = "pe_1_west_bind_0"} : !allo.stream<f32, 4>
    %28 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<f32, 4>
    %29 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<f32, 4>
    %30 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<f32, 4>
    %c1 = arith.constant 1 : index
    %31 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<f32, 4>
    %c0 = arith.constant 0 : index
    call @pe_west_load(%arg0, %c0, %31) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_west_load(%arg0, %c1, %30) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_north_load(%arg1, %c0, %29) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_north_load(%arg1, %c1, %28) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_1_west_load(%arg0, %c0, %27) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_1_west_load(%arg0, %c1, %26) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_1_north_load(%arg1, %c0, %25) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_1_north_load(%arg1, %c1, %24) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_2_west_load(%arg0, %c0, %23) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_2_west_load(%arg0, %c1, %22) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_2_north_load(%arg1, %c0, %21) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_2_north_load(%arg1, %c1, %20) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_3_west_load(%arg0, %c0, %19) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_3_west_load(%arg0, %c1, %18) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_3_north_load(%arg1, %c0, %17) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_3_north_load(%arg1, %c1, %16) : (memref<4x4xf32>, index, !allo.stream<f32, 4>) -> ()
    call @pe_r1(%arg2, %c0, %c0, %15, %29, %14, %31) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_r3(%arg2, %c0, %c1, %28, %13, %15) : (memref<4x4xf32>, index, index, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r0(%arg2, %c1, %c0, %12, %14, %30) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_r2(%arg2, %c1, %c1, %13, %12) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_1_r1(%arg2, %c0, %c0, %11, %25, %10, %27) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_1_r3(%arg2, %c0, %c1, %24, %9, %11) : (memref<4x4xf32>, index, index, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_1_r0(%arg2, %c1, %c0, %8, %10, %26) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_1_r2(%arg2, %c1, %c1, %9, %8) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_2_r1(%arg2, %c0, %c0, %7, %21, %6, %23) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_2_r3(%arg2, %c0, %c1, %20, %5, %7) : (memref<4x4xf32>, index, index, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_2_r0(%arg2, %c1, %c0, %4, %6, %22) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_2_r2(%arg2, %c1, %c1, %5, %4) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_3_r1(%arg2, %c0, %c0, %3, %17, %2, %19) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_3_r3(%arg2, %c0, %c1, %16, %1, %3) : (memref<4x4xf32>, index, index, !allo.stream<f32, 4>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_3_r0(%arg2, %c1, %c0, %0, %2, %18) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 4>) -> ()
    call @pe_3_r2(%arg2, %c1, %c1, %1, %0) : (memref<4x4xf32>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    return
  }
}
