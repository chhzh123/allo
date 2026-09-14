#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  func.func @pe_west_load(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg1, %arg3] {from = "local_A"} : memref<3x3xf32, #map>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 3> contains f32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_north_load(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "___", otypes = ""} {
    affine.for %arg3 = 0 to 3 {
      %0 = affine.load %arg0[%arg3, %arg1] {from = "local_B"} : memref<3x3xf32, #map>
      allo.stream_put(%arg2, [], %0) : !allo.stream<f32, 3> contains f32
    } {loop_name = "_t", op_name = "S__t_0", pipeline_ii = 1 : ui32}
    return
  }
  func.func @pe_r0(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<f32, 3> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r1(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<f32, 3> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
      %9 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r2(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
      %9 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r3(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r4(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 3>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 3>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<f32, 3> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
      %9 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r5(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 3>, %arg5: !allo.stream<f32, 2>, %arg6: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg7 = 0 to 3 {
      %1 = allo.stream_get(%arg6, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg4, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      allo.stream_put(%arg3, [], %8) : !allo.stream<f32, 2> contains f32
      %9 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg5, [], %9) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r6(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg4, [], %8) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r7(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 2>, %arg4: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "_____", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg5 = 0 to 3 {
      %1 = allo.stream_get(%arg4, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<f32, 2> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @pe_r8(%arg0: memref<3x3xf32, #map>, %arg1: index, %arg2: index, %arg3: !allo.stream<f32, 3>, %arg4: !allo.stream<f32, 2>, %arg5: !allo.stream<f32, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "acc"} : memref<f32>
    affine.store %cst, %alloc[] {to = "acc"} : memref<f32>
    affine.for %arg6 = 0 to 3 {
      %1 = allo.stream_get(%arg5, []) : !allo.stream<f32, 2> -> f32
      %alloc_0 = memref.alloc() {name = "a"} : memref<f32>
      affine.store %1, %alloc_0[] {to = "a"} : memref<f32>
      %2 = allo.stream_get(%arg3, []) : !allo.stream<f32, 3> -> f32
      %alloc_1 = memref.alloc() {name = "b"} : memref<f32>
      affine.store %2, %alloc_1[] {to = "b"} : memref<f32>
      %3 = affine.load %alloc_0[] {from = "a"} : memref<f32>
      %4 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %alloc[] {from = "acc"} : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloc[] {to = "acc"} : memref<f32>
      %8 = affine.load %alloc_1[] {from = "b"} : memref<f32>
      allo.stream_put(%arg4, [], %8) : !allo.stream<f32, 2> contains f32
    } {loop_name = "k", op_name = "S_k_0", pipeline_ii = 1 : ui32}
    %0 = affine.load %alloc[] {from = "acc"} : memref<f32>
    affine.store %0, %arg0[%arg1, %arg2] {to = "local_C"} : memref<3x3xf32, #map>
    return
  }
  func.func @top(%arg0: memref<3x3xf32, #map>, %arg1: memref<3x3xf32, #map>, %arg2: memref<3x3xf32, #map>) attributes {dataflow, itypes = "___", otypes = "", top} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %0 = allo.stream_construct() {name = "pe_east_west_2_2"} : !allo.stream<f32, 2>
    %1 = allo.stream_construct() {name = "pe_east_west_2_1"} : !allo.stream<f32, 2>
    %2 = allo.stream_construct() {name = "pe_south_north_2_2"} : !allo.stream<f32, 2>
    %3 = allo.stream_construct() {name = "pe_south_north_2_1"} : !allo.stream<f32, 2>
    %4 = allo.stream_construct() {name = "pe_east_west_1_2"} : !allo.stream<f32, 2>
    %5 = allo.stream_construct() {name = "pe_south_north_2_0"} : !allo.stream<f32, 2>
    %6 = allo.stream_construct() {name = "pe_east_west_1_1"} : !allo.stream<f32, 2>
    %7 = allo.stream_construct() {name = "pe_south_north_1_2"} : !allo.stream<f32, 2>
    %8 = allo.stream_construct() {name = "pe_south_north_1_1"} : !allo.stream<f32, 2>
    %9 = allo.stream_construct() {name = "pe_east_west_0_2"} : !allo.stream<f32, 2>
    %10 = allo.stream_construct() {name = "pe_south_north_1_0"} : !allo.stream<f32, 2>
    %11 = allo.stream_construct() {name = "pe_east_west_0_1"} : !allo.stream<f32, 2>
    %12 = allo.stream_construct() {name = "pe_north_bind_2"} : !allo.stream<f32, 3>
    %13 = allo.stream_construct() {name = "pe_north_bind_1"} : !allo.stream<f32, 3>
    %14 = allo.stream_construct() {name = "pe_north_bind_0"} : !allo.stream<f32, 3>
    %15 = allo.stream_construct() {name = "pe_west_bind_2"} : !allo.stream<f32, 3>
    %16 = allo.stream_construct() {name = "pe_west_bind_1"} : !allo.stream<f32, 3>
    %17 = allo.stream_construct() {name = "pe_west_bind_0"} : !allo.stream<f32, 3>
    call @pe_west_load(%arg0, %c0, %17) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_west_load(%arg0, %c1, %16) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_west_load(%arg0, %c2, %15) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_north_load(%arg1, %c0, %14) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_north_load(%arg1, %c1, %13) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_north_load(%arg1, %c2, %12) : (memref<3x3xf32, #map>, index, !allo.stream<f32, 3>) -> ()
    call @pe_r4(%arg2, %c0, %c0, %11, %14, %10, %17) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 3>, !allo.stream<f32, 2>, !allo.stream<f32, 3>) -> ()
    call @pe_r5(%arg2, %c0, %c1, %9, %13, %8, %11) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 3>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r8(%arg2, %c0, %c2, %12, %7, %9) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 3>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r1(%arg2, %c1, %c0, %6, %10, %5, %16) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 3>) -> ()
    call @pe_r2(%arg2, %c1, %c1, %4, %8, %3, %6) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r6(%arg2, %c1, %c2, %7, %2, %4) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r0(%arg2, %c2, %c0, %1, %5, %15) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 3>) -> ()
    call @pe_r3(%arg2, %c2, %c1, %0, %3, %1) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    call @pe_r7(%arg2, %c2, %c2, %2, %0) : (memref<3x3xf32, #map>, index, index, !allo.stream<f32, 2>, !allo.stream<f32, 2>) -> ()
    return
  }
}
