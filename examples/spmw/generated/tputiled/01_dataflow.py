import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: _T15[12, 4], W: _T15[4, 4, 2], Bias: int32[4], Prog: int32[12], Y: int32[6, 4]):
    tiled_mac_a_out_a_in: Stream[_T15, 2][4, 4]
    tiled_mac_p_out_p_in: Stream[int32, 2][4, 4]
    tiled_vpu_op_out_op_in: Stream[int32, 2][4]
    tiled_mac_a_in_bind: Stream[_T15, 2][4]
    tiled_vpu_z_in_bind: Stream[int32, 2][4]
    tiled_vpu_op_in_bind: Stream[int32, 2][1]
    tiled_vpu_y_out_bind: Stream[int32, 2][4]

    @df.kernel(mapping=[4], args=[A])
    def tiled_mac_a_in_load(local_A: _T15[12, 4]):
        _q0 = df.get_pid()
        for _t in range(12):
            tiled_mac_a_in_bind[_q0].put(local_A[_t, _q0])

    @df.kernel(mapping=[1], args=[Prog])
    def tiled_vpu_op_in_load(local_Prog: int32[12]):
        _q0 = df.get_pid()
        for _t in range(12):
            tiled_vpu_op_in_bind[_q0].put(local_Prog[_t])

    @df.kernel(mapping=[4, 4], args=[W])
    def tiled_mac(local_W: _T15[4, 4, 2]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_tiled_mac_16[_p0][_p1] == 0):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 1):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_vpu_z_in_bind[_p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 2):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = 0
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 3):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 4):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_in_bind[_p0].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 5):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_vpu_z_in_bind[_p1].put(p + a * wt)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 6):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_out_a_in[_p0, _p1].get()
                    p = 0
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
        with allo.meta_elif(_ROLE_tiled_mac_16[_p0][_p1] == 7):
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_in_bind[_p0].get()
                    p = tiled_mac_p_out_p_in[_p0, _p1].get()
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_vpu_z_in_bind[_p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_else():
            for m in range(MT):
                for t in range(NTILE):
                    a = tiled_mac_a_in_bind[_p0].get()
                    p = 0
                    wt: int32 = local_W[_p0, _p1, t]
                    tiled_mac_p_out_p_in[_p0 + 1, _p1].put(p + a * wt)
                    tiled_mac_a_out_a_in[_p0, _p1 + 1].put(a)

    @df.kernel(mapping=[4], args=[Bias])
    def tiled_vpu(local_Bias: int32[4]):
        _p0 = df.get_pid()
        with allo.meta_if(_ROLE_tiled_vpu_17[_p0] == 0):
            prog: int32[12] = 0
            for pc in range(NPROG_T):
                word: int32 = tiled_vpu_op_out_op_in[_p0].get()
                prog[pc] = word
                tiled_vpu_op_out_op_in[_p0 + 1].put(word)
            for m in range(MT):
                reg: int32[4] = 0
                for step in range(NPROG_T):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = tiled_vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0]
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
                    elif opcode == STORE:
                        tiled_vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_elif(_ROLE_tiled_vpu_17[_p0] == 1):
            prog: int32[12] = 0
            for pc in range(NPROG_T):
                word: int32 = tiled_vpu_op_out_op_in[_p0].get()
                prog[pc] = word
            for m in range(MT):
                reg: int32[4] = 0
                for step in range(NPROG_T):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = tiled_vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0]
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
                    elif opcode == STORE:
                        tiled_vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_else():
            prog: int32[12] = 0
            for pc in range(NPROG_T):
                word: int32 = tiled_vpu_op_in_bind[0].get()
                prog[pc] = word
                tiled_vpu_op_out_op_in[_p0 + 1].put(word)
            for m in range(MT):
                reg: int32[4] = 0
                for step in range(NPROG_T):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == ACCZ:
                        zz: int32 = tiled_vpu_z_in_bind[_p0].get()
                        reg[dst] = reg[dst] + zz
                    elif opcode == LOADB:
                        reg[dst] = local_Bias[_p0]
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
                    elif opcode == STORE:
                        tiled_vpu_y_out_bind[_p0].put(reg[dst])

    @df.kernel(mapping=[4], args=[Y])
    def tiled_vpu_y_out_drain(local_Y: int32[6, 4]):
        _q0 = df.get_pid()
        for _t in range(6):
            local_Y[_t, _q0] = tiled_vpu_y_out_bind[_q0].get()