# build_mac.tcl — build.tcl with only the ANN MAC coprocessor (rtl/mac_pcpi.v,
# custom-1) -> ./spike_top_mac.bit, for the cost table (the board runs use
# spike_top_both.bit). Usage (from vivado/):  vivado -mode batch -source build_mac.tcl
set mac_coproc 1
set bit_name spike_top_mac.bit
source build.tcl
