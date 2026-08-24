import allo
import allo.dataflow as df
from allo.ir.types import Stream

@df.region()
def top(Pr: int8[6, 8], V: int8[8, 2], Y: int8[6, 2]):
    mac_a_out_a_in: Stream[int8, 2][4, 4]
    mac_p_out_p_in: Stream[_T5, 2][14]
    mac_a_in_bind: Stream[int8, 2][8]
    act_z_in_bind: Stream[_T5, 2][2]
    act_y_out_bind: Stream[int8, 2][2]

    @df.kernel(mapping=[4, 2], args=[Pr])
    def mac_a_in_load(local_Pr: int8[6, 8]):
        _q0, _q1 = df.get_pid()
        for _t in range(6):
            mac_a_in_bind[_q0 * 2 + _q1].put(local_Pr[_t, _q1 * 2 // 2 * 4 + _q0])

    @df.kernel(mapping=[4, 4], args=[V])
    def mac(local_V: int8[8, 2]):
        _p0, _p1 = df.get_pid()
        with allo.meta_if(_ROLE_mac_14[_p0][_p1] == 0):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_CH_mac_p_out_p_in_p_in_6[_p0][_p1]].get()
                mac_p_out_p_in[_CH_mac_p_out_p_in_p_out_7[_p0][_p1]].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
        with allo.meta_elif(_ROLE_mac_14[_p0][_p1] == 1):
            for m in range(MT):
                a = mac_a_in_bind[_p0 * 2 + _p1 // 2].get()
                p = mac_p_out_p_in[_CH_mac_p_out_p_in_p_in_8[_p0][_p1]].get()
                mac_p_out_p_in[_CH_mac_p_out_p_in_p_out_9[_p0][_p1]].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_elif(_ROLE_mac_14[_p0][_p1] == 2):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = mac_p_out_p_in[_CH_mac_p_out_p_in_p_in_10[_p0][_p1]].get()
                act_z_in_bind[_p1 - 2].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
        with allo.meta_elif(_ROLE_mac_14[_p0][_p1] == 3):
            for m in range(MT):
                a = mac_a_out_a_in[_p0, _p1].get()
                p = 0
                mac_p_out_p_in[_CH_mac_p_out_p_in_p_out_11[_p0][_p1]].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
        with allo.meta_elif(_ROLE_mac_14[_p0][_p1] == 4):
            for m in range(MT):
                a = mac_a_in_bind[_p0 * 2 + _p1 // 2].get()
                p = mac_p_out_p_in[_CH_mac_p_out_p_in_p_in_12[_p0][_p1]].get()
                act_z_in_bind[_p1 - 2].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)
        with allo.meta_else():
            for m in range(MT):
                a = mac_a_in_bind[_p0 * 2 + _p1 // 2].get()
                p = 0
                mac_p_out_p_in[_CH_mac_p_out_p_in_p_out_13[_p0][_p1]].put(p + a * local_V[_p1 // 2 * 4 + _p0, _p1 % 2])
                mac_a_out_a_in[_p0, _p1 + 1].put(a)

    @df.kernel(mapping=[2])
    def act():
        _p0 = df.get_pid()
        for m in range(MT):
            z = act_z_in_bind[_p0].get()
            if z < 0:
                z = 0
            y: int8 = z >> SHIFT
            act_y_out_bind[_p0].put(y)

    @df.kernel(mapping=[2], args=[Y])
    def act_y_out_drain(local_Y: int8[6, 2]):
        _q0 = df.get_pid()
        for _t in range(6):
            local_Y[_t, _q0] = act_y_out_bind[_q0].get()