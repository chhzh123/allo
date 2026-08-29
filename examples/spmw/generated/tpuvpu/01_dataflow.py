import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: _T14[6, 4], W: _T14[4, 4], Bias: int32[4], Prog: int32[8], Y: int32[6, 4]):
    mac_a_out_a_in: Stream[_T14, 2][4, 4]
    mac_p_out_p_in: Stream[int32, 2][4, 4]
    vpu_op_out_op_in: Stream[int32, 2][4]
    mac_a_in_bind: Stream[_T14, 2][4]
    vpu_z_in_bind: Stream[int32, 2][4]
    vpu_op_in_bind: Stream[int32, 2][1]
    vpu_y_out_bind: Stream[int32, 2][4]

    @df.kernel(mapping=[4], args=[A])
    def mac_a_in_load(local_A: _T14[6, 4]):
        _q0 = df.get_pid()
        for _t in range(6):
            mac_a_in_bind[_q0].put(local_A[_t, _q0])

    @df.kernel(mapping=[1], args=[Prog])
    def vpu_op_in_load(local_Prog: int32[8]):
        _q0 = df.get_pid()
        for _t in range(8):
            vpu_op_in_bind[_q0].put(local_Prog[_t])

    @df.kernel(mapping=[4, 4], args=[W])
    def mac(local_W: _T14[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_mac_15[_p0][_p1] == 0):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 1):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                vpu_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 2):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 3):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 4):
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 5):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                vpu_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 6):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_15[_p0][_p1] == 7):
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                vpu_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_else():
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)

    @df.kernel(mapping=[4], args=[Bias])
    def vpu(local_Bias: int32[4]):
        _p0 = df.get_pid()
        with allo.meta_if(_ROLE_vpu_16[_p0] == 0):
            prog: int32[8] = 0
            for pc in range(NPROG):
                word: int32 = vpu_op_out_op_in[_p0].get()
                prog[pc] = word
                vpu_op_out_op_in[_p0 + 1].put(word)
            for m in range(MT):
                z: int32 = vpu_z_in_bind[_p0].get()
                reg: int32[4] = 0
                for step in range(NPROG):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == LOADZ:
                        reg[dst] = z
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
                        vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_elif(_ROLE_vpu_16[_p0] == 1):
            prog: int32[8] = 0
            for pc in range(NPROG):
                word: int32 = vpu_op_out_op_in[_p0].get()
                prog[pc] = word
            for m in range(MT):
                z: int32 = vpu_z_in_bind[_p0].get()
                reg: int32[4] = 0
                for step in range(NPROG):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == LOADZ:
                        reg[dst] = z
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
                        vpu_y_out_bind[_p0].put(reg[dst])
        with allo.meta_else():
            prog: int32[8] = 0
            for pc in range(NPROG):
                word: int32 = vpu_op_in_bind[0].get()
                prog[pc] = word
                vpu_op_out_op_in[_p0 + 1].put(word)
            for m in range(MT):
                z: int32 = vpu_z_in_bind[_p0].get()
                reg: int32[4] = 0
                for step in range(NPROG):
                    word2: int32 = prog[step]
                    opcode: int32 = word2 >> 24 & 255
                    dst: int32 = word2 >> 20 & 15
                    src: int32 = word2 >> 16 & 15
                    imm: int32 = word2 & 65535
                    if opcode == LOADZ:
                        reg[dst] = z
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
                        vpu_y_out_bind[_p0].put(reg[dst])

    @df.kernel(mapping=[4], args=[Y])
    def vpu_y_out_drain(local_Y: int32[6, 4]):
        _q0 = df.get_pid()
        for _t in range(6):
            local_Y[_t, _q0] = vpu_y_out_bind[_q0].get()