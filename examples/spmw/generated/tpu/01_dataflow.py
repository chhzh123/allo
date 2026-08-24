import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(A: int8[6, 4], W: int8[4, 4], Y: int8[6, 4]):
    mac_a_out_a_in: Stream[int8, 2][4, 4]
    mac_p_out_p_in: Stream[_T5, 2][4, 4]
    mac_a_in_bind: Stream[int8, 2][4]
    act_z_in_bind: Stream[_T5, 2][4]
    act_y_out_bind: Stream[int8, 2][4]

    @df.kernel(mapping=[4], args=[A])
    def mac_a_in_load(local_A: int8[6, 4]):
        _q0 = df.get_pid()
        for _t in range(6):
            mac_a_in_bind[_q0].put(local_A[_t, _q0])

    @df.kernel(mapping=[4, 4], args=[W])
    def mac(local_W: int8[4, 4]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_mac_6[_p0][_p1] == 0):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 1):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                act_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 2):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 3):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 4):
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 5):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                act_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 6):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
        with allo.meta_elif(_ROLE_mac_6[_p0][_p1] == 7):
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = mac_p_out_p_in[_p0, _p1].get()
                act_z_in_bind[_p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_else():
            for m in range(MT):
                a = mac_a_in_bind[_p0].get()
                p = 0
                mac_p_out_p_in[_p0 + 1, _p1].put(p + a * local_W[_p0, _p1])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)

    @df.kernel(mapping=[4])
    def act():
        _p0 = df.get_pid()
        for m in range(MT):
            z = act_z_in_bind[_p0].get()
            if z < 0:
                z = 0
            y: int8 = z >> SHIFT
            act_y_out_bind[_p0].put(y)

    @df.kernel(mapping=[4], args=[Y])
    def act_y_out_drain(local_Y: int8[6, 4]):
        _q0 = df.get_pid()
        for _t in range(6):
            local_Y[_t, _q0] = act_y_out_bind[_q0].get()