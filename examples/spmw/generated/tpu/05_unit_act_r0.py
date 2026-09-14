def act_r0_0():
    y_out: Stream[int8, 2][1]
    z_in: Stream[_T5, 2][1]
    for m in range(MT):
        z = z_in[0].get()
        if z < 0:
            z = 0
        y: int8 = z >> SHIFT
        y_out[0].put(y)