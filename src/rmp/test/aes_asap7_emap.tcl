source "helpers.tcl"

define_corners fast slow
read_liberty -corner slow ./asap7/asap7sc7p5t_AO_RVT_SS_nldm_211120.lib.gz
read_liberty -corner slow ./asap7/asap7sc7p5t_INVBUF_RVT_SS_nldm_220122.lib.gz
read_liberty -corner slow ./asap7/asap7sc7p5t_OA_RVT_SS_nldm_211120.lib.gz
read_liberty -corner slow ./asap7/asap7sc7p5t_SEQ_RVT_SS_nldm_220123.lib
read_liberty -corner slow ./asap7/asap7sc7p5t_SIMPLE_RVT_SS_nldm_211120.lib.gz
read_liberty -corner fast ./asap7/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz
read_liberty -corner fast ./asap7/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz
read_liberty -corner fast ./asap7/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz
read_liberty -corner fast ./asap7/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib
read_liberty -corner fast ./asap7/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz

read_lef ./asap7/asap7_tech_1x_201209.lef
read_lef ./asap7/asap7sc7p5t_28_R_1x_220121a.lef
read_verilog ./aes_asap7.v
link_design aes
# read_verilog ./sample.v
# link_design sample
read_sdc ./aes_asap7.sdc

set_layer_rc -layer M1 -resistance 7.04175E-02 -capacitance 1e-10
set_layer_rc -layer M2 -resistance 4.62311E-02 -capacitance 1.84542E-01
set_layer_rc -layer M3 -resistance 3.63251E-02 -capacitance 1.53955E-01
set_layer_rc -layer M4 -resistance 2.03083E-02 -capacitance 1.89434E-01
set_layer_rc -layer M5 -resistance 1.93005E-02 -capacitance 1.71593E-01
set_layer_rc -layer M6 -resistance 1.18619E-02 -capacitance 1.76146E-01
set_layer_rc -layer M7 -resistance 1.25311E-02 -capacitance 1.47030E-01
set_wire_rc -signal -resistance 3.23151E-02 -capacitance 1.73323E-01
set_wire_rc -clock -resistance 5.13971E-02 -capacitance 1.44549E-01

set_layer_rc -via V1 -resistance 1.72E-02
set_layer_rc -via V2 -resistance 1.72E-02
set_layer_rc -via V3 -resistance 1.72E-02
set_layer_rc -via V4 -resistance 1.18E-02
set_layer_rc -via V5 -resistance 1.18E-02
set_layer_rc -via V6 -resistance 8.20E-03
set_layer_rc -via V7 -resistance 8.20E-03
set_layer_rc -via V8 -resistance 6.30E-03

repair_timing

puts "-- Before --\n"
report_cell_usage
report_timing_histogram
report_checks
report_wns
report_tns

puts "-- After --\n"

emap -corner fast \
	-genlib_file ../../../third-party/mockturtle/experiments/cell_libraries/multioutput.genlib \
	-target timings \
	-map_multioutput \
	-verbose

read_sdc ./aes_asap7.sdc

write_verilog ./aes_asap7_emap.v
# write_verilog ./sample_emap.v

# estimate_parasitics -placement
# repair_timing

report_cell_usage
report_timing_histogram
report_cell_usage
report_checks
report_wns
report_tns

