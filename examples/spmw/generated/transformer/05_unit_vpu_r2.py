def vpu_r2_0():
    b: Stream[int32[2], 2][1]
    op_in: Stream[int32, 2][1]
    op_out: Stream[int32, 2][1]
    y_out: Stream[int32, 2][1]
    z_in: Stream[int32, 2][1]
    _st_b: int32[2] = b[0].get()
    header: int32 = op_in[0].get()
    op_out[0].put(header)
    plen: int32 = header & 65535
    nouts: int32 = header >> 16 & 65535
    prog: int32[16] = 0
    for pc in range(plen):
        word: int32 = op_in[0].get()
        prog[pc] = word
        op_out[0].put(word)
    for _pad in range(vprog_len - plen):
        spare: int32 = op_in[0].get()
        op_out[0].put(spare)
    denom: int32 = _st_b[1]
    rcp: int32 = 0
    if denom > 0:
        rcp = (1 << RCP_BITS) // denom
    r0: int32 = 0
    r1: int32 = 0
    r2: int32 = 0
    r3: int32 = 0
    pc2: int32 = 0
    for _k in range(nouts * plen):
        word2: int32 = prog[pc2]
        opcode: int32 = word2 >> 24 & 255
        dst: int32 = word2 >> 20 & 15
        src: int32 = word2 >> 16 & 15
        imm: int32 = word2 & 65535
        d: int32 = r0
        if dst == 1:
            d = r1
        elif dst == 2:
            d = r2
        elif dst == 3:
            d = r3
        a: int32 = r0
        if src == 1:
            a = r1
        elif src == 2:
            a = r2
        elif src == 3:
            a = r3
        wr: int32 = 1
        if opcode == ACCZ:
            zz: int32 = z_in[0].get()
            d = d + zz
        elif opcode == LOADZ:
            z2: int32 = z_in[0].get()
            d = z2
        elif opcode == LOADB:
            d = _st_b[src]
        elif opcode == LOADI:
            d = imm
        elif opcode == ADD:
            d = d + a
        elif opcode == MUL:
            d = d * a
        elif opcode == MAX:
            if a > d:
                d = a
        elif opcode == SHR:
            d = d >> imm
        elif opcode == SUB:
            d = d - a
        elif opcode == EXP2:
            e: int32 = d
            if e < 0:
                e = 0
            if e > 30:
                e = 30
            d = 1 << e
        elif opcode == LOADR:
            d = rcp
        elif opcode == STORE:
            y_out[0].put(d)
            wr = 0
        else:
            wr = 0
        if wr == 1:
            if dst == 0:
                r0 = d
            elif dst == 1:
                r1 = d
            elif dst == 2:
                r2 = d
            else:
                r3 = d
        pc2 = pc2 + 1
        if pc2 == plen:
            pc2 = 0