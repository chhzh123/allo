def pe_r2_0():
    c: Stream[int32, 2][1]
    east: Stream[int8, 2][1]
    north: Stream[int8, 2][1]
    west: Stream[int8, 2][1]
    acc: int32 = 0
    for k in range(n):
        a = west[0].get()
        b = north[0].get()
        acc += a * b
        east[0].put(a)
    c[0].put(acc)