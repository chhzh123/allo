def pe_r2_0():
    c: Stream[int32, 2][1]
    east: Stream[_T2, 2][1]
    north: Stream[_T2, 2][1]
    south: Stream[_T2, 2][1]
    west: Stream[_T2, 2][1]
    acc: int32 = 0
    for k in range(size):
        a = west[0].get()
        b = north[0].get()
        acc += a * b
        east[0].put(a)
        south[0].put(b)
    c[0].put(acc)