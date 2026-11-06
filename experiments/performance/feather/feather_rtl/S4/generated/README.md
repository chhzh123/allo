FEATHER's RTL is hand-written Verilog from the published design, so there is
no generation step: what Vivado reads is what is in `../source/` and in the
FEATHER repository it came from. This directory keeps the layout the same as
`spmw/`, where `generated/` holds what the compiler emitted.
