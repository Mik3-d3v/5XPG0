# build_snn.tcl — build.tcl with the SNN neuron coprocessor (rtl/snn_pcpi.v,
# custom-0 instructions) -> ./spike_top_snn.bit. The stock spike_top.bit is
# not touched. Usage (from vivado/):  vivado -mode batch -source build_snn.tcl
set snn_coproc 1
set bit_name spike_top_snn.bit
source build.tcl
