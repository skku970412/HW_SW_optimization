if { $argc < 5 } {
    puts "ERROR: Expected args: <root> <outdir> <part> <top> <clock_period_ns>"
    exit 2
}

set root            [lindex $argv 0]
set outdir          [lindex $argv 1]
set part            [lindex $argv 2]
set top             [lindex $argv 3]
set clock_period_ns [lindex $argv 4]

set rtl_dir [file join $root hw rtl]
set top_dir [file join $outdir $top]
file mkdir $top_dir

set srcs [list \
    [file join $rtl_dir gemm_core.sv] \
    [file join $rtl_dir attention_core.sv] \
    [file join $rtl_dir kv_cache.sv] \
    [file join $rtl_dir decoder_block_top.sv] \
    [file join $rtl_dir npu_top.sv] \
]

foreach src $srcs {
    if {![file exists $src]} {
        puts "ERROR: Missing RTL source: $src"
        exit 3
    }
}

read_verilog -sv $srcs

# For the KV-cache QoR point, we want deterministic BRAM usage.
# The portable RTL can still infer LUTRAM for small depths, so we allow
# forcing an XPM-backed implementation via a preprocessor define.
#
# NOTE: In Vivado Non-Project mode, use the synth_design -verilog_define option.
if { $top == "kv_cache" } {
    synth_design -top $top -part $part -verilog_define KV_CACHE_USE_XPM=1
} else {
    synth_design -top $top -part $part
}

if {[llength [get_ports clk]] > 0} {
    create_clock -name clk -period $clock_period_ns [get_ports clk]
}

report_utilization -file [file join $top_dir utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 -file [file join $top_dir timing_summary.rpt]
write_checkpoint -force [file join $top_dir post_synth.dcp]

set info_fp [open [file join $top_dir run_info.txt] "w"]
puts $info_fp "top=$top"
puts $info_fp "part=$part"
puts $info_fp "clock_period_ns=$clock_period_ns"
close $info_fp

puts "QOR_DONE top=$top part=$part clock=$clock_period_ns"
exit 0
