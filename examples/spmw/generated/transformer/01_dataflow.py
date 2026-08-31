import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: _T21[8, 4], W: _T21[4, 4, 4], Bias: int32[4, 2], MProg: int32[8, 4], VProg: int32[16], Y: int32[4, 4]):
    mac_a_out_a_in: Stream[_T21, 2][4, 4]
    mac_op_out_op_in: Stream[int32, 2][4, 4]
    mac_p_out_p_in: Stream[int32, 2][4, 4]
    vpu_op_out_op_in: Stream[int32, 2][4]
    mac_a_in_bind: Stream[_T21, 2][4]
    mac_op_in_bind: Stream[int32, 2][4]
    vpu_z_in_bind: Stream[int32, 2][4]
    vpu_op_in_bind: Stream[int32, 2][1]
    vpu_y_out_bind: Stream[int32, 2][4]

    @df.kernel(mapping=[4], args=[A])
    def mac_a_in_load(local_A: _T21[8, 4]):
        _q0 = df.get_pid()
        for _t in range(8):
            mac_a_in_bind[_q0].put(local_A[_t, _q0])

    @df.kernel(mapping=[4], args=[MProg])
    def mac_op_in_load(local_MProg: int32[8, 4]):
        _q0 = df.get_pid()
        for _t in range(8):
            mac_op_in_bind[_q0].put(local_MProg[_t, _q0])

    @df.kernel(mapping=[1], args=[VProg])
    def vpu_op_in_load(local_VProg: int32[16]):
        _q0 = df.get_pid()
        for _t in range(16):
            vpu_op_in_bind[_q0].put(local_VProg[_t])

    @df.kernel(mapping=[4, 4], args=[W])
    def mac(local_W: _T21[4, 4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_mac_22[_p0][_p1] == 0):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 1):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    vpu_z_in_bind[_p1].put(p + a * wt)
                elif opcode == MZERO:
                    vpu_z_in_bind[_p1].put(a * wt)
                else:
                    vpu_z_in_bind[_p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 2):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 3):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 4):
            for step in range(steps):
                word: int32 = mac_op_in_bind[_p0].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 5):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    vpu_z_in_bind[_p1].put(p + a * wt)
                elif opcode == MZERO:
                    vpu_z_in_bind[_p1].put(a * wt)
                else:
                    vpu_z_in_bind[_p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 6):
            for step in range(steps):
                word: int32 = mac_op_out_op_in[_p0, _p1].get()
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)
        with allo.meta_elif(_ROLE_mac_22[_p0][_p1] == 7):
            for step in range(steps):
                word: int32 = mac_op_in_bind[_p0].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    vpu_z_in_bind[_p1].put(p + a * wt)
                elif opcode == MZERO:
                    vpu_z_in_bind[_p1].put(a * wt)
                else:
                    vpu_z_in_bind[_p1].put(p)
        with allo.meta_else():
            for step in range(steps):
                word: int32 = mac_op_in_bind[_p0].get()
                mac_op_out_op_in[_p0, _p1 + 1].put(word)
                opcode: int32 = word >> 24 & 255
                tile: int32 = word >> 16 & 255
                a = mac_a_in_bind[_p0].get()
                p = 0
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
                wt: int32 = local_W[_p0, _p1, tile]
                if opcode == MACC:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                elif opcode == MZERO:
                    mac_p_out_p_in[_p0 + 1, _p1].put(a * wt)
                else:
                    mac_p_out_p_in[_p0 + 1, _p1].put(p)

    @df.kernel(mapping=[4], args=[Bias])
    def vpu(local_Bias: int32[4, 2]):
        _p0 = df.get_pid()
        with allo.meta_if(_ROLE_vpu_23[_p0] == 0):
            prog: int32[16] = 0
            for pc in range(vprog_len):
                word: int32 = vpu_op_out_op_in[_p0].get()
                prog[pc] = word
                vpu_op_out_op_in[_p0 + 1].put(word)
            reg: int32[4] = 0
            for m in range(outs):
                for pc2 in range(vprog_len):
                    word2: int32 = prog[pc2]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADZ:
                        z2: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = z2
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0, src]
                    elif opcode == LOADI:
                        reg[dst] = imm
                    elif opcode == ADD:
                        reg[dst] = reg[dst] + reg[src]
                    elif opcode == MUL:
                        reg[dst] = reg[dst] * reg[src]
                    elif opcode == MAX:
                        if reg[src] > reg[dst]:
                            reg[dst] = reg[src]
                    elif opcode == SHR:
                        reg[dst] = reg[dst] >> imm
                    elif opcode == SUB:
                        reg[dst] = reg[dst] - reg[src]
                    elif opcode == EXP2:
                        e: int32 = reg[dst]
                        if e < 0:
                            e = 0
                        if e > 30:
                            e = 30
                        reg[dst] = 1 << e
                    elif opcode == RECIP:
                        d: int32 = reg[dst]
                        if d > 0:
                            reg[dst] = (1 << imm) // d
                        else:
                            reg[dst] = 0
                    elif opcode == STORE:
                        vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_elif(_ROLE_vpu_23[_p0] == 1):
            prog: int32[16] = 0
            for pc in range(vprog_len):
                word: int32 = vpu_op_out_op_in[_p0].get()
                prog[pc] = word
            reg: int32[4] = 0
            for m in range(outs):
                for pc2 in range(vprog_len):
                    word2: int32 = prog[pc2]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADZ:
                        z2: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = z2
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0, src]
                    elif opcode == LOADI:
                        reg[dst] = imm
                    elif opcode == ADD:
                        reg[dst] = reg[dst] + reg[src]
                    elif opcode == MUL:
                        reg[dst] = reg[dst] * reg[src]
                    elif opcode == MAX:
                        if reg[src] > reg[dst]:
                            reg[dst] = reg[src]
                    elif opcode == SHR:
                        reg[dst] = reg[dst] >> imm
                    elif opcode == SUB:
                        reg[dst] = reg[dst] - reg[src]
                    elif opcode == EXP2:
                        e: int32 = reg[dst]
                        if e < 0:
                            e = 0
                        if e > 30:
                            e = 30
                        reg[dst] = 1 << e
                    elif opcode == RECIP:
                        d: int32 = reg[dst]
                        if d > 0:
                            reg[dst] = (1 << imm) // d
                        else:
                            reg[dst] = 0
                    elif opcode == STORE:
                        vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_else():
            prog: int32[16] = 0
            for pc in range(vprog_len):
                word: int32 = vpu_op_in_bind[0].get()
                prog[pc] = word
                vpu_op_out_op_in[_p0 + 1].put(word)
            reg: int32[4] = 0
            for m in range(outs):
                for pc2 in range(vprog_len):
                    word2: int32 = prog[pc2]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADZ:
                        z2: int32 = vpu_z_in_bind[_p0].get()
                        reg[dst] = z2
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0, src]
                    elif opcode == LOADI:
                        reg[dst] = imm
                    elif opcode == ADD:
                        reg[dst] = reg[dst] + reg[src]
                    elif opcode == MUL:
                        reg[dst] = reg[dst] * reg[src]
                    elif opcode == MAX:
                        if reg[src] > reg[dst]:
                            reg[dst] = reg[src]
                    elif opcode == SHR:
                        reg[dst] = reg[dst] >> imm
                    elif opcode == SUB:
                        reg[dst] = reg[dst] - reg[src]
                    elif opcode == EXP2:
                        e: int32 = reg[dst]
                        if e < 0:
                            e = 0
                        if e > 30:
                            e = 30
                        reg[dst] = 1 << e
                    elif opcode == RECIP:
                        d: int32 = reg[dst]
                        if d > 0:
                            reg[dst] = (1 << imm) // d
                        else:
                            reg[dst] = 0
                    elif opcode == STORE:
                        vpu_y_out_bind[_p0].put(reg[dst])

    @df.kernel(mapping=[4], args=[Y])
    def vpu_y_out_drain(local_Y: int32[4, 4]):
        _q0 = df.get_pid()
        for _t in range(4):
            local_Y[_t, _q0] = vpu_y_out_bind[_q0].get()