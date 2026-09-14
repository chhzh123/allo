def feed_3_up_load_io_0(local_Bt: int8[4, 4]):
    chan: Stream[int8[4], 16][1]
    _q0 = 0
    for _t in range(4):
        _blk: int8[4] = 0
        for _b0 in range(4):
            _blk[_b0] = local_Bt[_t, _b0]
        chan[0].put(_blk)