#map = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
module {
  memref.global "private" @_tab0 : memref<4x1x1xi32> = dense<[[[0]], [[2]], [[1]], [[3]]]>
  func.func @bfly_up_in_load(%arg0: memref<8x2xf32, #map>, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @_tab0 : memref<4x1x1xi32>
    affine.for %arg3 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg4 = 0 to 2 {
        affine.store %cst, %alloc[%arg4] : memref<2xf32>
      }
      affine.for %arg4 = 0 to 2 {
        %1 = affine.load %0[%arg1, %arg3, 0] {from = "_tab0"} : memref<4x1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg4] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg4] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<2xf32>, 4> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_tab0_0 : memref<4x1x1xi32> = dense<[[[4]], [[6]], [[5]], [[7]]]>
  func.func @bfly_lo_in_load(%arg0: memref<8x2xf32, #map>, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @_tab0_0 : memref<4x1x1xi32>
    affine.for %arg3 = 0 to 1 {
      %alloc = memref.alloc() {name = "_blk"} : memref<2xf32>
      affine.for %arg4 = 0 to 2 {
        affine.store %cst, %alloc[%arg4] : memref<2xf32>
      }
      affine.for %arg4 = 0 to 2 {
        %1 = affine.load %0[%arg1, %arg3, 0] {from = "_tab0"} : memref<4x1x1xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg0[%2, %arg4] {from = "local_X"} : memref<8x2xf32, #map>
        affine.store %3, %alloc[%arg4] {to = "_blk"} : memref<2xf32>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
      allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<2xf32>, 4> contains memref<2xf32>
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_st_tw : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_r0(%arg0: index, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 2>, %arg4: !allo.stream<memref<2xf32>, 2>, %arg5: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw : memref<4x2xf32>
    %1 = arith.index_cast %arg0 {name = "%1"} : index to i32
    %2 = arith.shli %c1_i32, %1 {name = "%2"} : i32
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %2, %alloc[] {to = "span"} : memref<i32>
    %3 = affine.load %alloc[] {from = "span"} : memref<i32>
    %4 = arith.index_cast %arg1 {name = "%4"} : index to i33
    %5 = arith.extsi %3 {name = "%5"} : i32 to i33
    %6 = arith.remsi %4, %5 {name = "%6"} : i33
    %7 = arith.floordivsi %c4_i32, %3 {name = "%7"} : i32
    %8 = arith.extsi %6 {name = "%8"} : i33 to i65
    %9 = arith.extsi %7 {name = "%9"} : i32 to i65
    %10 = arith.muli %8, %9 {name = "%10"} : i65
    %alloc_0 = memref.alloc() {name = "k"} : memref<i65>
    affine.store %10, %alloc_0[] {to = "k"} : memref<i65>
    %11 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %12 = arith.index_cast %11 {name = "%12"} : i65 to index
    %13 = memref.load %0[%12, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %13, %alloc_1[] {to = "wr"} : memref<f32>
    %14 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %15 = arith.index_cast %14 {name = "%16"} : i65 to index
    %16 = memref.load %0[%15, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %16, %alloc_2[] {to = "wi"} : memref<f32>
    %17 = allo.stream_get(%arg4, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %18 = allo.stream_get(%arg2, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %19 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %20 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %21 = arith.mulf %19, %20 {name = "%23"} : f32
    %22 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %23 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%26"} : f32
    %25 = arith.subf %21, %24 {name = "%27"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %25, %alloc_3[] {to = "tr"} : memref<f32>
    %26 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %27 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %28 = arith.mulf %26, %27 {name = "%30"} : f32
    %29 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %30 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %31 = arith.mulf %29, %30 {name = "%33"} : f32
    %32 = arith.addf %28, %31 {name = "%34"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %32, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg6] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg6] : memref<2xf32>
    }
    %33 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %34 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %35 = arith.addf %33, %34 {name = "%38"} : f32
    affine.store %35, %alloc_5[0] {to = "u"} : memref<2xf32>
    %36 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %37 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %38 = arith.addf %36, %37 {name = "%41"} : f32
    affine.store %38, %alloc_5[1] {to = "u"} : memref<2xf32>
    %39 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %40 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %41 = arith.subf %39, %40 {name = "%44"} : f32
    affine.store %41, %alloc_6[0] {to = "l"} : memref<2xf32>
    %42 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %43 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %44 = arith.subf %42, %43 {name = "%47"} : f32
    affine.store %44, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg5, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_0 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_r1(%arg0: index, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 2>, %arg3: !allo.stream<memref<2xf32>, 4>, %arg4: !allo.stream<memref<2xf32>, 2>, %arg5: !allo.stream<memref<2xf32>, 4>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_0 : memref<4x2xf32>
    %1 = arith.index_cast %arg0 {name = "%1"} : index to i32
    %2 = arith.shli %c1_i32, %1 {name = "%2"} : i32
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %2, %alloc[] {to = "span"} : memref<i32>
    %3 = affine.load %alloc[] {from = "span"} : memref<i32>
    %4 = arith.index_cast %arg1 {name = "%4"} : index to i33
    %5 = arith.extsi %3 {name = "%5"} : i32 to i33
    %6 = arith.remsi %4, %5 {name = "%6"} : i33
    %7 = arith.floordivsi %c4_i32, %3 {name = "%7"} : i32
    %8 = arith.extsi %6 {name = "%8"} : i33 to i65
    %9 = arith.extsi %7 {name = "%9"} : i32 to i65
    %10 = arith.muli %8, %9 {name = "%10"} : i65
    %alloc_0 = memref.alloc() {name = "k"} : memref<i65>
    affine.store %10, %alloc_0[] {to = "k"} : memref<i65>
    %11 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %12 = arith.index_cast %11 {name = "%12"} : i65 to index
    %13 = memref.load %0[%12, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %13, %alloc_1[] {to = "wr"} : memref<f32>
    %14 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %15 = arith.index_cast %14 {name = "%16"} : i65 to index
    %16 = memref.load %0[%15, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %16, %alloc_2[] {to = "wi"} : memref<f32>
    %17 = allo.stream_get(%arg4, []) {name = "a"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %18 = allo.stream_get(%arg2, []) {name = "c"} : !allo.stream<memref<2xf32>, 2> -> memref<2xf32>
    %19 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %20 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %21 = arith.mulf %19, %20 {name = "%23"} : f32
    %22 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %23 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%26"} : f32
    %25 = arith.subf %21, %24 {name = "%27"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %25, %alloc_3[] {to = "tr"} : memref<f32>
    %26 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %27 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %28 = arith.mulf %26, %27 {name = "%30"} : f32
    %29 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %30 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %31 = arith.mulf %29, %30 {name = "%33"} : f32
    %32 = arith.addf %28, %31 {name = "%34"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %32, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg6] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg6] : memref<2xf32>
    }
    %33 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %34 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %35 = arith.addf %33, %34 {name = "%38"} : f32
    affine.store %35, %alloc_5[0] {to = "u"} : memref<2xf32>
    %36 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %37 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %38 = arith.addf %36, %37 {name = "%41"} : f32
    affine.store %38, %alloc_5[1] {to = "u"} : memref<2xf32>
    %39 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %40 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %41 = arith.subf %39, %40 {name = "%44"} : f32
    affine.store %41, %alloc_6[0] {to = "l"} : memref<2xf32>
    %42 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %43 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %44 = arith.subf %42, %43 {name = "%47"} : f32
    affine.store %44, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg5, [], %alloc_5) : !allo.stream<memref<2xf32>, 4> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 4> contains memref<2xf32>
    return
  }
  memref.global "private" @_st_tw_1 : memref<4x2xf32> = dense<[[1.000000e+00, -0.000000e+00], [0.707106769, -0.707106769], [6.12323426E-17, -1.000000e+00], [-0.707106769, -0.707106769]]>
  func.func @bfly_r2(%arg0: index, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 4>, %arg3: !allo.stream<memref<2xf32>, 2>, %arg4: !allo.stream<memref<2xf32>, 4>, %arg5: !allo.stream<memref<2xf32>, 2>) attributes {df.kernel, itypes = "______", otypes = ""} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant {name = "%c4_i32"} 4 : i32
    %c1_i32 = arith.constant {name = "%c1_i32"} 1 : i32
    %0 = memref.get_global @_st_tw_1 : memref<4x2xf32>
    %1 = arith.index_cast %arg0 {name = "%1"} : index to i32
    %2 = arith.shli %c1_i32, %1 {name = "%2"} : i32
    %alloc = memref.alloc() {name = "span"} : memref<i32>
    affine.store %2, %alloc[] {to = "span"} : memref<i32>
    %3 = affine.load %alloc[] {from = "span"} : memref<i32>
    %4 = arith.index_cast %arg1 {name = "%4"} : index to i33
    %5 = arith.extsi %3 {name = "%5"} : i32 to i33
    %6 = arith.remsi %4, %5 {name = "%6"} : i33
    %7 = arith.floordivsi %c4_i32, %3 {name = "%7"} : i32
    %8 = arith.extsi %6 {name = "%8"} : i33 to i65
    %9 = arith.extsi %7 {name = "%9"} : i32 to i65
    %10 = arith.muli %8, %9 {name = "%10"} : i65
    %alloc_0 = memref.alloc() {name = "k"} : memref<i65>
    affine.store %10, %alloc_0[] {to = "k"} : memref<i65>
    %11 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %12 = arith.index_cast %11 {name = "%12"} : i65 to index
    %13 = memref.load %0[%12, %c0] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_1 = memref.alloc() {name = "wr"} : memref<f32>
    affine.store %13, %alloc_1[] {to = "wr"} : memref<f32>
    %14 = affine.load %alloc_0[] {from = "k"} : memref<i65>
    %15 = arith.index_cast %14 {name = "%16"} : i65 to index
    %16 = memref.load %0[%15, %c1] {from = "_st_tw"} : memref<4x2xf32>
    %alloc_2 = memref.alloc() {name = "wi"} : memref<f32>
    affine.store %16, %alloc_2[] {to = "wi"} : memref<f32>
    %17 = allo.stream_get(%arg4, []) {name = "a"} : !allo.stream<memref<2xf32>, 4> -> memref<2xf32>
    %18 = allo.stream_get(%arg2, []) {name = "c"} : !allo.stream<memref<2xf32>, 4> -> memref<2xf32>
    %19 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %20 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %21 = arith.mulf %19, %20 {name = "%23"} : f32
    %22 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %23 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %24 = arith.mulf %22, %23 {name = "%26"} : f32
    %25 = arith.subf %21, %24 {name = "%27"} : f32
    %alloc_3 = memref.alloc() {name = "tr"} : memref<f32>
    affine.store %25, %alloc_3[] {to = "tr"} : memref<f32>
    %26 = affine.load %alloc_1[] {from = "wr"} : memref<f32>
    %27 = affine.load %18[1] {from = "c"} : memref<2xf32>
    %28 = arith.mulf %26, %27 {name = "%30"} : f32
    %29 = affine.load %alloc_2[] {from = "wi"} : memref<f32>
    %30 = affine.load %18[0] {from = "c"} : memref<2xf32>
    %31 = arith.mulf %29, %30 {name = "%33"} : f32
    %32 = arith.addf %28, %31 {name = "%34"} : f32
    %alloc_4 = memref.alloc() {name = "ti"} : memref<f32>
    affine.store %32, %alloc_4[] {to = "ti"} : memref<f32>
    %alloc_5 = memref.alloc() {name = "u"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_5[%arg6] : memref<2xf32>
    }
    %alloc_6 = memref.alloc() {name = "l"} : memref<2xf32>
    affine.for %arg6 = 0 to 2 {
      affine.store %cst, %alloc_6[%arg6] : memref<2xf32>
    }
    %33 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %34 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %35 = arith.addf %33, %34 {name = "%38"} : f32
    affine.store %35, %alloc_5[0] {to = "u"} : memref<2xf32>
    %36 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %37 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %38 = arith.addf %36, %37 {name = "%41"} : f32
    affine.store %38, %alloc_5[1] {to = "u"} : memref<2xf32>
    %39 = affine.load %17[0] {from = "a"} : memref<2xf32>
    %40 = affine.load %alloc_3[] {from = "tr"} : memref<f32>
    %41 = arith.subf %39, %40 {name = "%44"} : f32
    affine.store %41, %alloc_6[0] {to = "l"} : memref<2xf32>
    %42 = affine.load %17[1] {from = "a"} : memref<2xf32>
    %43 = affine.load %alloc_4[] {from = "ti"} : memref<f32>
    %44 = arith.subf %42, %43 {name = "%47"} : f32
    affine.store %44, %alloc_6[1] {to = "l"} : memref<2xf32>
    allo.stream_put(%arg5, [], %alloc_5) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    allo.stream_put(%arg3, [], %alloc_6) : !allo.stream<memref<2xf32>, 2> contains memref<2xf32>
    return
  }
  memref.global "private" @_tab0_1 : memref<4x1x1xi32> = dense<[[[0]], [[1]], [[2]], [[3]]]>
  func.func @bfly_up_out_drain(%arg0: memref<8x2xf32, #map>, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    %0 = memref.get_global @_tab0_1 : memref<4x1x1xi32>
    affine.for %arg3 = 0 to 1 {
      %1 = allo.stream_get(%arg2, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 4> -> memref<2xf32>
      affine.for %arg4 = 0 to 2 {
        %2 = affine.load %1[%arg4] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg1, %arg3, 0] {from = "_tab0"} : memref<4x1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg4] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  memref.global "private" @_tab0_2 : memref<4x1x1xi32> = dense<[[[4]], [[5]], [[6]], [[7]]]>
  func.func @bfly_lo_out_drain(%arg0: memref<8x2xf32, #map>, %arg1: index, %arg2: !allo.stream<memref<2xf32>, 4>) attributes {df.kernel, itypes = "___", otypes = ""} {
    %0 = memref.get_global @_tab0_2 : memref<4x1x1xi32>
    affine.for %arg3 = 0 to 1 {
      %1 = allo.stream_get(%arg2, []) {name = "_blk"} : !allo.stream<memref<2xf32>, 4> -> memref<2xf32>
      affine.for %arg4 = 0 to 2 {
        %2 = affine.load %1[%arg4] {from = "_blk"} : memref<2xf32>
        %3 = affine.load %0[%arg1, %arg3, 0] {from = "_tab0"} : memref<4x1x1xi32>
        %4 = arith.index_cast %3 : i32 to index
        memref.store %2, %arg0[%4, %arg4] {to = "local_Y"} : memref<8x2xf32, #map>
      } {loop_name = "_b0", op_name = "S__b0_0", pipeline_ii = 1 : ui32}
    } {loop_name = "_t", op_name = "S__t_0"}
    return
  }
  func.func @top(%arg0: memref<8x2xf32, #map>, %arg1: memref<8x2xf32, #map>) attributes {dataflow, itypes = "__", otypes = "", top} {
    %c0 = arith.constant {name = "%c0"} 0 : index
    %c1 = arith.constant {name = "%c1"} 1 : index
    %c2 = arith.constant {name = "%c2"} 2 : index
    %c3 = arith.constant {name = "%c3"} 3 : index
    %0 = allo.stream_construct() {name = "bfly_up_out_bind_3"} : !allo.stream<memref<2xf32>, 4>
    %1 = allo.stream_construct() {name = "bfly_lo_out_bind_3"} : !allo.stream<memref<2xf32>, 4>
    %2 = allo.stream_construct() {name = "bfly_up_out_bind_2"} : !allo.stream<memref<2xf32>, 4>
    %3 = allo.stream_construct() {name = "bfly_lo_out_bind_2"} : !allo.stream<memref<2xf32>, 4>
    %4 = allo.stream_construct() {name = "bfly_up_out_bind_1"} : !allo.stream<memref<2xf32>, 4>
    %5 = allo.stream_construct() {name = "bfly_lo_out_bind_1"} : !allo.stream<memref<2xf32>, 4>
    %6 = allo.stream_construct() {name = "bfly_up_out_bind_0"} : !allo.stream<memref<2xf32>, 4>
    %7 = allo.stream_construct() {name = "bfly_lo_out_bind_0"} : !allo.stream<memref<2xf32>, 4>
    %8 = allo.stream_construct() {name = "bfly_key_13"} : !allo.stream<memref<2xf32>, 2>
    %9 = allo.stream_construct() {name = "bfly_key_15"} : !allo.stream<memref<2xf32>, 2>
    %10 = allo.stream_construct() {name = "bfly_key_12"} : !allo.stream<memref<2xf32>, 2>
    %11 = allo.stream_construct() {name = "bfly_key_14"} : !allo.stream<memref<2xf32>, 2>
    %12 = allo.stream_construct() {name = "bfly_key_9"} : !allo.stream<memref<2xf32>, 2>
    %13 = allo.stream_construct() {name = "bfly_key_11"} : !allo.stream<memref<2xf32>, 2>
    %14 = allo.stream_construct() {name = "bfly_key_8"} : !allo.stream<memref<2xf32>, 2>
    %15 = allo.stream_construct() {name = "bfly_key_10"} : !allo.stream<memref<2xf32>, 2>
    %16 = allo.stream_construct() {name = "bfly_key_6"} : !allo.stream<memref<2xf32>, 2>
    %17 = allo.stream_construct() {name = "bfly_key_7"} : !allo.stream<memref<2xf32>, 2>
    %18 = allo.stream_construct() {name = "bfly_key_4"} : !allo.stream<memref<2xf32>, 2>
    %19 = allo.stream_construct() {name = "bfly_key_5"} : !allo.stream<memref<2xf32>, 2>
    %20 = allo.stream_construct() {name = "bfly_key_2"} : !allo.stream<memref<2xf32>, 2>
    %21 = allo.stream_construct() {name = "bfly_key_3"} : !allo.stream<memref<2xf32>, 2>
    %22 = allo.stream_construct() {name = "bfly_key_0"} : !allo.stream<memref<2xf32>, 2>
    %23 = allo.stream_construct() {name = "bfly_key_1"} : !allo.stream<memref<2xf32>, 2>
    %24 = allo.stream_construct() {name = "bfly_lo_in_bind_3"} : !allo.stream<memref<2xf32>, 4>
    %25 = allo.stream_construct() {name = "bfly_lo_in_bind_2"} : !allo.stream<memref<2xf32>, 4>
    %26 = allo.stream_construct() {name = "bfly_lo_in_bind_1"} : !allo.stream<memref<2xf32>, 4>
    %27 = allo.stream_construct() {name = "bfly_lo_in_bind_0"} : !allo.stream<memref<2xf32>, 4>
    %28 = allo.stream_construct() {name = "bfly_up_in_bind_3"} : !allo.stream<memref<2xf32>, 4>
    %29 = allo.stream_construct() {name = "bfly_up_in_bind_2"} : !allo.stream<memref<2xf32>, 4>
    %30 = allo.stream_construct() {name = "bfly_up_in_bind_1"} : !allo.stream<memref<2xf32>, 4>
    %31 = allo.stream_construct() {name = "bfly_up_in_bind_0"} : !allo.stream<memref<2xf32>, 4>
    call @bfly_up_in_load(%arg0, %c0, %31) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_in_load(%arg0, %c1, %30) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_in_load(%arg0, %c2, %29) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_in_load(%arg0, %c3, %28) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_in_load(%arg0, %c0, %27) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_in_load(%arg0, %c1, %26) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_in_load(%arg0, %c2, %25) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_in_load(%arg0, %c3, %24) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_r2(%c0, %c0, %27, %23, %31, %22) : (index, index, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r2(%c0, %c1, %26, %21, %30, %20) : (index, index, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r2(%c0, %c2, %25, %19, %29, %18) : (index, index, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r2(%c0, %c3, %24, %17, %28, %16) : (index, index, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r0(%c1, %c0, %20, %15, %22, %14) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r0(%c1, %c1, %21, %13, %23, %12) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r0(%c1, %c2, %16, %11, %18, %10) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r0(%c1, %c3, %17, %9, %19, %8) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 2>) -> ()
    call @bfly_r1(%c2, %c0, %10, %7, %14, %6) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_r1(%c2, %c1, %8, %5, %12, %4) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_r1(%c2, %c2, %11, %3, %15, %2) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_r1(%c2, %c3, %9, %1, %13, %0) : (index, index, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>, !allo.stream<memref<2xf32>, 2>, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_out_drain(%arg1, %c0, %6) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_out_drain(%arg1, %c1, %4) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_out_drain(%arg1, %c2, %2) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_up_out_drain(%arg1, %c3, %0) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_out_drain(%arg1, %c0, %7) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_out_drain(%arg1, %c1, %5) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_out_drain(%arg1, %c2, %3) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    call @bfly_lo_out_drain(%arg1, %c3, %1) : (memref<8x2xf32, #map>, index, !allo.stream<memref<2xf32>, 4>) -> ()
    return
  }
}
