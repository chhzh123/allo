def mac_r2_0():
    w: Stream[_T14, 2][1]
    a_in: Stream[_T14, 2][1]
    a_out: Stream[_T14, 2][1]
    p_out: Stream[int32, 2][1]
    _st_w: _T14 = w[0].get()
    for m in range(MT):
        a = a_in[0].get()
        p = 0
        p_out[0].put(p + a * _st_w)
        a_out[0].put(a)