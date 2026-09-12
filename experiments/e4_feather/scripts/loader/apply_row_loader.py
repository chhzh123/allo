"""E4: the row-wise weight loader, on top of the corrected controller.

    apply_row_loader.py <RTL dir>

FEATHER's weight port is already DPE_COL_NUM bytes wide and every column
already carries its own valid:

    feather_top.v: WEIGHTS_DATA_WIDTH = 8*DPE_COL_NUM
                   w_dpe_weights[0][COL]       = <bus>[COL*8 +: 8]
                   w_dpe_weights_valid[0][COL] = <valids>[COL]
    feather_controller.v: w_weights_valid_from_ctrl_to_dpe = ~0 in a feed state

so all DPE_COL_NUM valids are asserted every feed cycle and every column is
offered its own byte. What throws DPE_COL_NUM-1 of them away is the select:
feather_pe.v stores only when i_pe_sel equals its *global* id
DPE_ROW_NUM*col + row, which spans 0..N^2-1, and the controller free-runs
pe_sel through all N^2 of them one a cycle. One PE stores per cycle, so a
WEIGHTS_DEPTH = N deep file in each of N^2 PEs costs N^3 cycles.

This patch makes pe_sel name a *row* of the array instead of one PE:

  - the controller wraps r_pe_sel at WEIGHTS_DEPTH-1 (a sweep is N cycles);
  - the PE matches on the row field of its id, THIS_PE_ID % WEIGHTS_DEPTH.

Every column's PE in that row then takes its own byte in the same cycle, and a
full load is N^2 cycles. Nothing in the datapath moves: the weights, pe_sel and
the ping/pong select descend the same per-column daisy chain a row a cycle, so
PE row r still meets its own pe_sel value alongside its own weight byte, and
the compute path, the reduction network and the arithmetic are untouched.

The SRAM image has to be re-packed to match -- N distinct weights per word
instead of one -- which is e4_feather_gen.py's `--loader row`.
"""

import sys

PE_OLD = """    integer i;
"""

PE_NEW = """    //  [E4 row-wise loader] pe_sel names a row of the array, not one PE.
    //  THIS_PE_ID is the global DPE_ROW_NUM*col + row, so matching it admitted
    //  exactly one PE per cycle and the other DPE_COL_NUM-1 bytes of the weight
    //  word -- which the top already slices out per column, each with its own
    //  valid -- were dropped. Matching the row field alone admits the whole row
    //  at once. WEIGHTS_DEPTH is DPE_ROW_NUM, so THIS_PE_ID % WEIGHTS_DEPTH is
    //  this PE's row. The weights, pe_sel and the ping/pong select all descend
    //  the same daisy chain a row a cycle, so row r still meets its own pe_sel
    //  value in the same cycle as its own weight byte.
    localparam [PE_SEL_WIDTH-1:0] THIS_PE_SEL_MATCH = THIS_PE_ID % WEIGHTS_DEPTH;

    integer i;
"""

PE_GATE_OLD = """                if(i_pe_sel == THIS_PE_ID)
"""

PE_GATE_NEW = """                if(i_pe_sel == THIS_PE_SEL_MATCH)
"""

CTRL_OLD = """                        r_weights_pingpong_rd_addr          <=  r_weights_pingpong_rd_addr + 1;
                        r_weights_to_use                    <=  $unsigned(WEIGHTS_DEPTH)-1;
                        r_pe_sel                            <=  r_pe_sel + 1;
"""

CTRL_NEW = """                        r_weights_pingpong_rd_addr          <=  r_weights_pingpong_rd_addr + 1;
                        r_weights_to_use                    <=  $unsigned(WEIGHTS_DEPTH)-1;
                        // [E4 row-wise loader] pe_sel wraps at DPE_ROW_NUM
                        // (= WEIGHTS_DEPTH) instead of running free through all
                        // DPE_COL_NUM*DPE_ROW_NUM ids, so it selects a row of the
                        // array rather than one PE and every column's PE in that
                        // row stores its own byte of the
                        // WEIGHTS_SRAM_DATA_WIDTH-wide word in the same cycle.
                        // A sweep is DPE_ROW_NUM cycles, so a WEIGHTS_DEPTH-deep
                        // load of the whole array is N^2 cycles, not N^3.
                        if(r_pe_sel == $unsigned(WEIGHTS_DEPTH)-1)
                        begin
                            r_pe_sel                        <=  0;
                        end
                        else
                        begin
                            r_pe_sel                        <=  r_pe_sel + 1;
                        end
"""


def patch(path, pairs):
    src = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert src.count(old) == 1, (path, src.count(old), old.strip()[:60])
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print("patched", path)


def main():
    rtl = sys.argv[1].rstrip("/")
    ctrl = f"{rtl}/feather_controller.v"
    assert "[E4 corrected variant]" in open(ctrl, encoding="utf-8").read(), (
        "apply_patch.py (the corrected controller) must be applied first"
    )
    patch(f"{rtl}/feather_pe.v", [(PE_OLD, PE_NEW), (PE_GATE_OLD, PE_GATE_NEW)])
    patch(ctrl, [(CTRL_OLD, CTRL_NEW)])


if __name__ == "__main__":
    main()
