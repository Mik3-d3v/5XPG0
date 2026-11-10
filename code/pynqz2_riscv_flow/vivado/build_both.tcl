# build_both.tcl — build.tcl with the SNN neuron coprocessor (rtl/snn_pcpi.v,
# custom-0) AND the ANN MAC coprocessor (rtl/mac_pcpi.v, custom-1) on the one
# PCPI port -> ./spike_top_both.bit. spike_top.bit and spike_top_snn.bit are
# not touched. Usage (from vivado/):  vivado -mode batch -source build_both.tcl
set snn_coproc 1
set mac_coproc 1
set bit_name spike_top_both.bit
source build.tcl
