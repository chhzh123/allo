import sys
path = sys.argv[1]
src = open(path).read()
old = """            if(r_pe_sel == $unsigned(DPE_COL_NUM*WEIGHTS_DEPTH)-1)
            begin
                r_weights_ping_pong_sel <=  ~r_weights_ping_pong_sel;
            end
"""
new = """            // [E4 corrected variant] The PEs' ping/pong select toggles once per
            // complete feed -- the read address reaching addr_end in a feed
            // state -- instead of once per sweep of DPE_COL_NUM*WEIGHTS_DEPTH
            // cycles. pe_sel visits every PE once a sweep and a PE stores one
            // weight per visit, so the shipped toggle put a PE's even-index
            // weights in one buffer and its odd-index ones in the other, and the
            // compute, which reads one buffer, saw half of every file. A feed of
            // WEIGHTS_DEPTH sweeps now fills one buffer completely, the select
            // flips as it ends, and the next feed fills the other while the
            // compute reads this one.
            if(((r_weights_buf_ping_pong_state == WEIGHTS_PINGPONG_PING_FEED_DPE) ||
                (r_weights_buf_ping_pong_state == WEIGHTS_PINGPONG_PONG_FEED_DPE)) &&
               (r_weights_pingpong_rd_addr == i_weights_write_addr_end))
            begin
                r_weights_ping_pong_sel <=  ~r_weights_ping_pong_sel;
            end
"""
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new))
print("patched", path)
