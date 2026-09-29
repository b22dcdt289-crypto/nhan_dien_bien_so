# Drive the on-board character ROM through JTAG in-system source/probe.
# This script records *actual FPGA outputs*; its CSV is local-only.
package require ::quartus::jtag
package require ::quartus::insystem_source_probe

set project_dir [file dirname [file normalize [info script]]]
set output_csv [file join $project_dir generated board_results.csv]
set hardware [lindex [get_hardware_names] 0]
if {$hardware eq ""} { error "No local JTAG hardware found" }
set device [lindex [get_device_names -hardware_name $hardware] 0]
if {$device eq ""} { error "No FPGA device found on $hardware" }
set instances [get_insystem_source_probe_instance_info -hardware_name $hardware -device_name $device]
puts "HARDWARE=$hardware"
puts "DEVICE=$device"
puts "INSTANCES=$instances"
if {[llength $instances] != 1} { error "Expected one OCR in-system source/probe" }

start_insystem_source_probe -hardware_name $hardware -device_name $device
set previous_probe [read_probe_data -instance_index 0 -value_in_hex]
set previous_raw [expr "0x$previous_probe"]
set previous_seq [expr {($previous_raw >> 24) & 0xff}]
set fp [open $output_csv w]
puts $fp "sample_index,predicted_class,cycles,result_seq,probe_hex"
set failed 0
for {set index 0} {$index < 50} {incr index} {
    set ready 0
    for {set start_attempt 0} {$start_attempt < 8 && !$ready} {incr start_attempt} {
        # A readback and generous low interval make the JTAG/50MHz-domain
        # handshake deterministic even when the USB-Blaster buffers writes.
        write_source_data -instance_index 0 -value [format %02x $index] -value_in_hex
        set source_low [read_source_data -instance_index 0 -value_in_hex]
        after 100
        write_source_data -instance_index 0 -value [format %02x [expr {$index | 0x80}]] -value_in_hex
        set source_high [read_source_data -instance_index 0 -value_in_hex]
        after 500
        set probe [read_probe_data -instance_index 0 -value_in_hex]
        set raw [expr "0x$probe"]
        set got_index [expr {($raw >> 8) & 0x7f}]
        set valid [expr {($raw >> 7) & 1}]
        set seq [expr {($raw >> 24) & 0xff}]
        if {$valid && $got_index == $index && $seq != $previous_seq} {
            after 30
            set confirm [read_probe_data -instance_index 0 -value_in_hex]
            if {$confirm eq $probe} {
                set ready 1
            } else {
                puts "UNSTABLE sample=$index first=$probe second=$confirm"
            }
        } else {
            puts "RETRY sample=$index try=$start_attempt low=$source_low high=$source_high probe=$probe"
        }
    }
    write_source_data -instance_index 0 -value [format %02x $index] -value_in_hex
    if {!$ready} {
        set failed 1
        puts "TIMEOUT sample=$index probe=$probe"
        break
    }
    set class_id [expr {$raw & 0x1f}]
    set cycles [expr {($raw >> 32) & 0xffffffff}]
    puts $fp "$index,$class_id,$cycles,$seq,$probe"
    flush $fp
    set previous_seq $seq
    puts "SAMPLE=$index CLASS=$class_id CYCLES=$cycles"
}
close $fp
end_insystem_source_probe
if {$failed} { error "FPGA OCR demo did not finish every sample" }
puts "RESULT_CSV=$output_csv"
