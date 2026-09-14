def pe_r2_0():
    c: Stream[float32, 2][1]
    north: Stream[float32, 2][1]
    west: Stream[float32, 2][1]
    acc: float32 = 0
    for k in range(K):
        a = west[0].get()
        b = north[0].get()
        acc += a * b
    c[0].put(acc)