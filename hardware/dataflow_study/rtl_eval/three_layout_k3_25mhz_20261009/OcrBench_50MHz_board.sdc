# DE10-Lite P11 is driven by the board's 50 MHz oscillator (20 ns period).
# The OCR state machine uses a synchronous /2 clock enable, not a 25 MHz clock.
# Keep every register-to-register path constrained to the physical 50 MHz clock.
create_clock -name MAX10_CLK1_50 -period 20.000 [get_ports {MAX10_CLK1_50}]
derive_clock_uncertainty

# LEDR0 is a human-visible status LED, not an externally clocked data output.
set_false_path -to [get_ports {LEDR0}]

# These dedicated JTAG pins belong to the USB-Blaster/TAP, not the OCR data
# interface. Quartus constrains the internal JTAG clock separately as TCK.
# Only the board-level pin paths are excluded; internal TCK paths remain timed.
set_false_path -from [get_ports {altera_reserved_tdi altera_reserved_tms}]
set_false_path -to [get_ports {altera_reserved_tdo}]
