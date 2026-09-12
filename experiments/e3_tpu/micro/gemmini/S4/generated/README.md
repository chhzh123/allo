# Elaborated Verilog is not committed here

`MxuVpu.v` at S=4 is megabytes of Chisel output -- 2.4 MB at 16x16 -- and it
is one command away:

    cd <gemmini project>
    MESH_DIM=4 SCALE_MODE=shift sbt -batch "runMain gen.ElaborateMxuVpu"

which writes `mxuvpu_out_4_shift/MxuVpu.v`. That is the netlist
`../../../gemmini/source/mxuvpu_pnr.sh` routes and
`../../../gemmini/report/S4/` reports on.
