#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  memref.global "private" @_ix0 : memref<1x1xi32> = dense<0>
  func.func @bfly_up_in_load_0(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_0 : memref<1x1xi32> = dense<2>
  func.func @bfly_up_in_load_1(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_0 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_1 : memref<1x1xi32> = dense<1>
  func.func @bfly_up_in_load_2(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_1 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_2 : memref<1x1xi32> = dense<3>
  func.func @bfly_up_in_load_3(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_2 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_3 : memref<1x1xi32> = dense<4>
  func.func @bfly_lo_in_load_0(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_3 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_4 : memref<1x1xi32> = dense<6>
  func.func @bfly_lo_in_load_1(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_4 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_5 : memref<1x1xi32> = dense<5>
  func.func @bfly_lo_in_load_2(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_5 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_6 : memref<1x1xi32> = dense<7>
  func.func @bfly_lo_in_load_3(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %0 = memref.get_global @_ix0_6 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        affine.store %cst, %alloc[%arg3] : memref<2xf32>
      }
      affine.for %arg3 = 0 to 2 {
        %1 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg3] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg3] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg1, [], %alloc) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_st_tw : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_0_0(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c1_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c0_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_0 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_0_1(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_0 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c1_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c1_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_1 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_0_2(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_1 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c1_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c2_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_2 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_0_3(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_2 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c1_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c3_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_3 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_1_0(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %0 = memref.get_global @_st_tw_3 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c2_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c0_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_4 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_1_1(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_4 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c2_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c1_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_5 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_1_2(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %0 = memref.get_global @_st_tw_5 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c2_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c2_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_6 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_1_3(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %0 = memref.get_global @_st_tw_6 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c2_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c3_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_7 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_2_0(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c0_i32 = arith.constant {name = "%c0_i32"} 0 : i32
    %0 = memref.get_global @_st_tw_7 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c4_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c0_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_8 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_2_1(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_8 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c4_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c1_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_9 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_2_2(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c2_i32 = arith.constant {name = "%c2_i32"} 2 : i32
    %0 = memref.get_global @_st_tw_9 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c4_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c2_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_10 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_2_3(%arg0: !allo.stream<memref<2xf32>, 2>, %arg1: !allo.stream<memref<2xf32>, 2>, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iioo"} {
    %cst = arith.constant {name = "%cst"} 0.000000e+00 : f32
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c3_i32 = arith.constant {name = "%c3_i32"} 3 : i32
    %0 = memref.get_global @_st_tw_10 : memref<4x2xf32>
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %c4_i32, %alloc[] {to = "span"} : memref<i32>
    %1 = affine.load %alloc[] {from = "span"} : memref<i32>
    %2 = arith.remsi %c3_i32, %1 {name = "%2"} : i32
    %3 = arith.floordivsi %c4_i32, %1 {name = "%3"} : i32
    %4 = arith.extsi %2 {name = "%4"} : i32 to i64
    %5 = arith.extsi %3 {name = "%5"} : i32 to i64
    %6 = arith.muli %4, %5 {name = "%6"} : i64
    %alloc_0 = memref.alloc() {name = "k"} : memref<i64>
    affine.store %6, %alloc_0[] {to = "k"} : memref<i64>
    %7 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %8 = arith.index_cast %7 {name = "%8"} : i64 to index
    %9 = memref.load %0[%8, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %9, %alloc_1[] {to = "wr"} : memref<f32>
    %10 = affine.load %alloc_0[] {from = "k"} : memref<i64>
    %11 = arith.index_cast %10 {name = "%11"} : i64 to index
    %12 = memref.load %0[%11, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %12, %alloc_2[] {to = "wi"} : memref<f32>
    %13 = allo.stream_get(%arg0, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %14 = allo.stream_get(%arg1, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %15 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %16 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %17 = arith.mulf %15, %16 {name = "%17"} : f32
    %18 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %19 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %20 = arith.mulf %18, %19 {name = "%20"} : f32
    %21 = arith.subf %17, %20 {name = "%21"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %21, %alloc_3[] {to = "tr"} : memref<f32>
    %22 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %23 = affine.load %14[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%24"} : f32
    %25 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %26 = affine.load %14[0] {from = "c"} : memref<2xf32>
    %27 = arith.mulf %25, %26 {name = "%27"} : f32
    %28 = arith.addf %24, %27 {name = "%28"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %28, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg4] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg4 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg4] : memref<2xf32>
    }
    %29 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %30 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %31 = arith.addf %29, %30 {name = "%31"} : f32
    affine.store %31, %alloc_5[0] {to = "u"} : memref<2xf32>
    %32 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %33 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %34 = arith.addf %32, %33 {name = "%34"} : f32
    affine.store %34, %alloc_5[1] {to = "u"} : memref<2xf32>
    %35 = affine.load %13[0] {from = "a"} : memref<2xf32>
    %36 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %37 = arith.subf %35, %36 {name = "%37"} : f32
    affine.store %37, %alloc_6[0] {to = "l"} : memref<2xf32>
    %38 = affine.load %13[1] {from = "a"} : memref<2xf32>
    %39 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %40 = arith.subf %38, %39 {name = "%40"} : f32
    affine.store %40, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg2, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_ix0_7 : memref<1x1xi32> = dense<0>
  func.func @bfly_up_out_drain_0(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_7 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_8 : memref<1x1xi32> = dense<1>
  func.func @bfly_up_out_drain_1(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_8 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_9 : memref<1x1xi32> = dense<2>
  func.func @bfly_up_out_drain_2(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_9 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_10 : memref<1x1xi32> = dense<3>
  func.func @bfly_up_out_drain_3(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_10 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_11 : memref<1x1xi32> = dense<4>
  func.func @bfly_lo_out_drain_0(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_11 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_12 : memref<1x1xi32> = dense<5>
  func.func @bfly_lo_out_drain_1(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_12 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_13 : memref<1x1xi32> = dense<6>
  func.func @bfly_lo_out_drain_2(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_13 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_ix0_14 : memref<1x1xi32> = dense<7>
  func.func @bfly_lo_out_drain_3(%arg0: memref<8x2xf32, #map>, %arg1: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_i"} {
    %0 = memref.get_global @_ix0_14 : memref<1x1xi32>
    affine.for %arg2 = 0 to 1 {
      %1 = allo.stream_get(%arg1, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
      affine.for %arg3 = 0 to 2 {
        %2 = affine.load %1[%arg3] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg2, 0] {from = "_ix0"} : memref<1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg3] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<8x2xf32, #map>, %arg1: memref<8x2xf32, #map>) attributes {dataflow, itypes = "__", top} {
    %0 = allo.stream_construct() {name = "bfly_key_0"} : !allo.stream<memref<2xf32>, 2>
    %1 = allo.stream_construct() {name = "bfly_key_1"} : !allo.stream<memref<2xf32>, 2>
    %2 = allo.stream_construct() {name = "bfly_key_2"} : !allo.stream<memref<2xf32>, 2>
    %3 = allo.stream_construct() {name = "bfly_key_3"} : !allo.stream<memref<2xf32>, 2>
    %4 = allo.stream_construct() {name = "bfly_key_4"} : !allo.stream<memref<2xf32>, 2>
    %5 = allo.stream_construct() {name = "bfly_key_5"} : !allo.stream<memref<2xf32>, 2>
    %6 = allo.stream_construct() {name = "bfly_key_6"} : !allo.stream<memref<2xf32>, 2>
    %7 = allo.stream_construct() {name = "bfly_key_7"} : !allo.stream<memref<2xf32>, 2>
    %8 = allo.stream_construct() {name = "bfly_key_8"} : !allo.stream<memref<2xf32>, 2>
    %9 = allo.stream_construct() {name = "bfly_key_9"} : !allo.stream<memref<2xf32>, 2>
    %10 = allo.stream_construct() {name = "bfly_key_10"} : !allo.stream<memref<2xf32>, 2>
    %11 = allo.stream_construct() {name = "bfly_key_11"} : !allo.stream<memref<2xf32>, 2>
    %12 = allo.stream_construct() {name = "bfly_key_12"} : !allo.stream<memref<2xf32>, 2>
    %13 = allo.stream_construct() {name = "bfly_key_13"} : !allo.stream<memref<2xf32>, 2>
    %14 = allo.stream_construct() {name = "bfly_key_14"} : !allo.stream<memref<2xf32>, 2>
    %15 = allo.stream_construct() {name = "bfly_key_15"} : !allo.stream<memref<2xf32>, 2>
    %16 = allo.stream_construct() {name = "bfly_up_in_bind_0"} : !allo.stream<memref<2xf32>, 2>
    %17 = allo.stream_construct() {name = "bfly_up_in_bind_1"} : !allo.stream<memref<2xf32>, 2>
    %18 = allo.stream_construct() {name = "bfly_up_in_bind_2"} : !allo.stream<memref<2xf32>, 2>
    %19 = allo.stream_construct() {name = "bfly_up_in_bind_3"} : !allo.stream<memref<2xf32>, 2>
    %20 = allo.stream_construct() {name = "bfly_lo_in_bind_0"} : !allo.stream<memref<2xf32>, 2>
    %21 = allo.stream_construct() {name = "bfly_lo_in_bind_1"} : !allo.stream<memref<2xf32>, 2>
    %22 = allo.stream_construct() {name = "bfly_lo_in_bind_2"} : !allo.stream<memref<2xf32>, 2>
    %23 = allo.stream_construct() {name = "bfly_lo_in_bind_3"} : !allo.stream<memref<2xf32>, 2>
    %24 = allo.stream_construct() {name = "bfly_up_out_bind_0"} : !allo.stream<memref<2xf32>, 2>
    %25 = allo.stream_construct() {name = "bfly_up_out_bind_1"} : !allo.stream<memref<2xf32>, 2>
    %26 = allo.stream_construct() {name = "bfly_up_out_bind_2"} : !allo.stream<memref<2xf32>, 2>
    %27 = allo.stream_construct() {name = "bfly_up_out_bind_3"} : !allo.stream<memref<2xf32>, 2>
    %28 = allo.stream_construct() {name = "bfly_lo_out_bind_0"} : !allo.stream<memref<2xf32>, 2>
    %29 = allo.stream_construct() {name = "bfly_lo_out_bind_1"} : !allo.stream<memref<2xf32>, 2>
    %30 = allo.stream_construct() {name = "bfly_lo_out_bind_2"} : !allo.stream<memref<2xf32>, 2>
    %31 = allo.stream_construct() {name = "bfly_lo_out_bind_3"} : !allo.stream<memref<2xf32>, 2>
    call @bfly_up_in_load_0(%arg0, %16) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_in_load_1(%arg0, %17) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_in_load_2(%arg0, %18) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_in_load_3(%arg0, %19) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_in_load_0(%arg0, %20) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_in_load_1(%arg0, %21) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_in_load_2(%arg0, %22) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_in_load_3(%arg0, %23) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_0_0(%16, %20, %0, %1) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_0_1(%17, %21, %2, %3) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_0_2(%18, %22, %4, %5) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_0_3(%19, %23, %6, %7) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_1_0(%0, %2, %8, %10) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_1_1(%1, %3, %9, %11) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_1_2(%4, %6, %12, %14) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_1_3(%5, %7, %13, %15) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_2_0(%8, %12, %24, %28) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_2_1(%9, %13, %25, %29) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_2_2(%10, %14, %26, %30) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_2_3(%11, %15, %27, %31) : (!allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_out_drain_0(%arg1, %24) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_out_drain_1(%arg1, %25) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_out_drain_2(%arg1, %26) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_up_out_drain_3(%arg1, %27) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_out_drain_0(%arg1, %28) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_out_drain_1(%arg1, %29) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_out_drain_2(%arg1, %30) : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_lo_out_drain_3(%arg1, %31) {last} : (memref<8x2xf32, #map>, !allo.stream<memref<2xf32>, 2>) -> ()
    return
  }
}
