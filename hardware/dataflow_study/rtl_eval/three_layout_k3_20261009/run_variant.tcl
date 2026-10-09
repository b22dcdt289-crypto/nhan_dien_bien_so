# Read actual OCR outputs over USB-Blaster/JTAG. For stream variants, send every
# 32x32 INT8 character as 64 successive 16-byte chunks before starting the MAC.
package require ::quartus::jtag
package require ::quartus::insystem_source_probe

if {[llength $argv] < 1 || [llength $argv] > 2} {
    error "Usage: quartus_stp -t run_variant.tcl <variant> ?sample_limit?"
}
set variant [lindex $argv 0]
if {$variant ni {dense_rom dense_stream pruned_rom pruned_stream}} {
    error "Unknown variant: $variant"
}
set streaming [string match *_stream $variant]
set root [file dirname [file normalize [info script]]]
set generated [file join $root $variant generated]
set csv_path [file join $generated board_results.csv]
set image_path [file join $generated images.hex]
set fh [open $image_path r]
set image_bytes [split [string trim [read $fh]] "\n"]
close $fh
if {[llength $image_bytes] % 1024 != 0} { error "Image ROM length is not a multiple of 1024" }
set sample_count [expr {[llength $image_bytes] / 1024}]
if {[llength $argv] == 2} {
    set sample_limit [lindex $argv 1]
    if {![string is integer -strict $sample_limit] || $sample_limit < 1 || $sample_limit > $sample_count} {
        error "sample_limit must be from 1 to $sample_count"
    }
    set sample_count $sample_limit
}
if {$sample_count > 127} { error "Sample count exceeds 7-bit JTAG sample ID" }

set candidates [lsearch -all -inline -glob [get_hardware_names] "USB-Blaster*"]
if {[llength $candidates] != 1} { error "Expected exactly one local USB-Blaster: $candidates" }
set hardware [lindex $candidates 0]
set devices [get_device_names -hardware_name $hardware]
if {[llength $devices] != 1} { error "Expected one JTAG device: $devices" }
set device [lindex $devices 0]
set instances [get_insystem_source_probe_instance_info -hardware_name $hardware -device_name $device]
if {[llength $instances] != 1} { error "Expected one OCR source/probe: $instances" }
set width [lindex [lindex $instances 0] 1]
set expected_width [expr {$streaming ? 144 : 8}]
if {$width != $expected_width} { error "Wrong bitstream for $variant: source width $width expected $expected_width" }
puts "HARDWARE=$hardware DEVICE=$device VARIANT=$variant SOURCE_WIDTH=$width SAMPLES=$sample_count"

start_insystem_source_probe -hardware_name $hardware -device_name $device
set previous_raw [expr "0x[read_probe_data -instance_index 0 -value_in_hex]"]
set previous_seq [expr {($previous_raw >> 24) & 0xff}]
set output [open $csv_path w]
puts $output "sample_index,predicted_class,cycles,result_seq,jtag_load_ms,jtag_total_ms,probe_hex"
set toggle 0
set failed 0
for {set sample 0} {$sample < $sample_count} {incr sample} {
    set sample_start [clock milliseconds]
    set load_ms 0
    if {$streaming} {
        # Clear previous start while keeping the previous toggle value.
        set low_control [expr {($sample << 8) | ($toggle << 6)}]
        write_source_data -instance_index 0 -value [format "%04x%032x" $low_control 0] -value_in_hex
        read_source_data -instance_index 0 -value_in_hex
        after 10
        set load_start [clock milliseconds]
        for {set chunk 0} {$chunk < 64} {incr chunk} {
            set toggle [expr {!$toggle}]
            set payload ""
            for {set byte 15} {$byte >= 0} {incr byte -1} {
                append payload [lindex $image_bytes [expr {$sample * 1024 + $chunk * 16 + $byte}]]
            }
            set control [expr {($sample << 8) | ($toggle << 6) | $chunk}]
            set data [format "%04x%s" $control $payload]
            write_source_data -instance_index 0 -value $data -value_in_hex
            read_source_data -instance_index 0 -value_in_hex
            set acknowledged 0
            for {set attempt 0} {$attempt < 40} {incr attempt} {
                after 2
                set probe [read_probe_data -instance_index 0 -value_in_hex]
                set raw [expr "0x$probe"]
                set loaded [expr {($raw >> 16) & 0x7f}]
                if {$loaded == $chunk + 1} {
                    set acknowledged 1
                    break
                }
            }
            if {!$acknowledged} {
                puts "LOAD_TIMEOUT sample=$sample chunk=$chunk loaded=$loaded probe=$probe"
                set failed 1
                break
            }
        }
        if {$failed} { break }
        set load_ms [expr {[clock milliseconds] - $load_start}]
        set start_control [expr {($sample << 8) | (1 << 7) | ($toggle << 6)}]
        write_source_data -instance_index 0 -value [format "%04x%032x" $start_control 0] -value_in_hex
        read_source_data -instance_index 0 -value_in_hex
    } else {
        write_source_data -instance_index 0 -value [format %02x $sample] -value_in_hex
        read_source_data -instance_index 0 -value_in_hex
        after 100
        write_source_data -instance_index 0 -value [format %02x [expr {$sample | 0x80}]] -value_in_hex
        read_source_data -instance_index 0 -value_in_hex
        after 500
    }
    set ready 0
    for {set attempt 0} {$attempt < 40} {incr attempt} {
        after 5
        set probe [read_probe_data -instance_index 0 -value_in_hex]
        set raw [expr "0x$probe"]
        set got_sample [expr {($raw >> 8) & 0x7f}]
        set valid [expr {($raw >> 7) & 1}]
        set seq [expr {($raw >> 24) & 0xff}]
        if {$valid && $got_sample == $sample && $seq != $previous_seq} {
            after 10
            if {[read_probe_data -instance_index 0 -value_in_hex] eq $probe} {
                set ready 1
                break
            }
        }
    }
    if {!$ready} {
        puts "RESULT_TIMEOUT sample=$sample probe=$probe"
        set failed 1
        break
    }
    if {$seq != (($previous_seq + 1) & 0xff)} {
        puts "RESULT_SEQUENCE_ERROR sample=$sample previous=$previous_seq current=$seq probe=$probe"
        set failed 1
        break
    }
    set class_id [expr {$raw & 0x1f}]
    set cycles [expr {($raw >> 32) & 0xffffffff}]
    set total_ms [expr {[clock milliseconds] - $sample_start}]
    puts $output "$sample,$class_id,$cycles,$seq,$load_ms,$total_ms,$probe"
    flush $output
    set previous_seq $seq
    puts "SAMPLE=$sample CLASS=$class_id CYCLES=$cycles LOAD_MS=$load_ms TOTAL_MS=$total_ms"
}
close $output
end_insystem_source_probe
if {$failed} { error "Incomplete FPGA benchmark: $variant" }
puts "RESULT_CSV=$csv_path"
