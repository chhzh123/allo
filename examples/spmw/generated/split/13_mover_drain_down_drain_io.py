def drain_down_drain_io_0(local_Ct: int32[4, 4]):
    chan: Stream[int32, 4][1]
    _pid0: Stream[int32, 1][1]
    _q0: int32 = _pid0[0].get()
    for _t in range(4):
        local_Ct[_q0, _t] = chan[0].get()