FEATHER's RTL is hand-written Verilog (`github.com/maeri-project/FEATHER`,
`FEATHER_RTL/RTL` at 225951b), so there is no generation step. Vivado read that
RTL with the patches in `../../../scripts/` applied; the patched trees are not
in the repository. This directory keeps the layout the same as `spmw/`, where
`generated/` holds what the compiler emitted.
