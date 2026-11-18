# ESP32-S3 to DE10-Lite OCR UART

## Physical connection

| ESP32-S3 N16R8 | DE10-Lite | Direction |
|---|---|---|
| GPIO43 / TX | JP1 physical pin 1, V10 / GPIO_[0] | ESP32 to FPGA `UART_RX` |
| GPIO44 / RX | JP1 physical pin 3, V9 / GPIO_[2] | FPGA `UART_TX` to ESP32 |
| GND | JP1 physical pin 12, GND | Common reference |

Use 3.3 V logic. Do not connect either board's 5 V pin to a GPIO. UART is 8-N-1. The ESP32 uses 1,843,200 baud; the FPGA divides its 50 MHz clock by 27, giving approximately 1,851,852 baud (about 0.46% difference).

## Processing sequence

1. The OV3660 captures a frame. ESP32 applies the configured ROI, grayscale conversion, Otsu thresholding, connected-component segmentation, left-to-right ordering, and per-glyph normalization to 28×28.
2. ESP32 sends each of the eight glyphs as a 790-byte packet, waiting for the matching FPGA response before sending the next glyph.
3. DE10-Lite validates the request, runs the INT8 LeNet classifier, converts its logits to Q15 probabilities, selects a confidence bin, and applies the frozen validation confusion posterior and blend weight.
4. DE10-Lite returns the raw class, calibrated top-3 classes and scores, confusion-table support, per-character accept/reject, and FPGA cycle count. It also evaluates the whole one-row plate on glyph 7 using the seven earlier FPGA character decisions.
5. ESP32 validates response CRC and frame/glyph IDs, assembles the eight FPGA predictions into the plate string, logs the result over USB CDC, and updates the camera web UI. ESP32 performs no classifier calibration or acceptance-threshold decision.

The confusion matrix is an offline calibration artifact only. No expected character is sent with an inference request, and no runtime label is available to the FPGA.

## Request packet: 790 bytes

| Offset | Width | Field |
|---:|---:|---|
| 0 | 1 | Sync `0xA5` |
| 1 | 1 | Version `0x01` |
| 2 | 1 | Frame ID |
| 3 | 1 | Glyph index, 0–7 left to right |
| 4–787 | 784 | Row-major grayscale pixels, 28×28, unsigned 0–255 |
| 788–789 | 2 | CRC-16/CCITT-FALSE, high byte first |

The request CRC covers bytes 0–787. CRC parameters: initial value `0xFFFF`, polynomial `0x1021`, no reflection, no final XOR.

## FPGA response packet: 25 bytes

| Offset | Width | Field |
|---:|---:|---|
| 0–1 | 2 | Sync `0x5A 0xA5` |
| 2 | 1 | Frame ID |
| 3 | 1 | Glyph index |
| 4 | 1 | Transport/inference status; `0` means a valid inference result |
| 5 | 1 | Bit 0: character accepted; bit 1: whole plate accepted (meaningful for glyph 7) |
| 6 | 1 | Raw LeNet argmax class index |
| 7 | 1 | Confidence bin, 0–5 |
| 8–9 | 2 | Validation support for the raw class and confidence bin, little-endian |
| 10 | 1 | Calibrated top-1 class index |
| 11–12 | 2 | Calibrated top-1 probability, unsigned Q0.15, little-endian |
| 13 | 1 | Calibrated top-2 class index |
| 14–15 | 2 | Calibrated top-2 probability, unsigned Q0.15, little-endian |
| 16 | 1 | Calibrated top-3 class index |
| 17–18 | 2 | Calibrated top-3 probability, unsigned Q0.15, little-endian |
| 19–22 | 4 | FPGA inference cycles, little-endian |
| 23–24 | 2 | CRC-16/CCITT-FALSE over bytes 2–22, high byte first |

Class indices use `0123456789ABCDEFGHKLMNPSTUVXYZ`. The FPGA uses an exponential LUT for temperature 0.1, Q15 normalization, Jeffreys-smoothed `P(true | raw class, confidence bin)`, and a maximum 30% validation-matrix blend. A glyph is accepted at calibrated top-1 probability ≥0.72 and top-1/top-2 margin ≥0.30; a changed class also requires matrix support ≥20. A plate is accepted by the FPGA only if all eight glyph decisions are accepted.

## Timing and measurement

- LeNet character latency is `cycles / 50` microseconds at 50 MHz. The calibration FSM adds approximately 1,000 normalization cycles plus table lookup/ranking cycles; compiled and measured counts must be read from the active build and packet.
- UART round-trip time includes 790 request bytes, FPGA inference/postprocessing, and 25 response bytes.
- End-to-end camera time includes capture, decode, segmentation, eight sequential FPGA transactions, and web-result assembly.
- Synthetic validation metrics and live-camera measurements are reported separately. Camera frames are not used as training labels.

## Hardware bench status — 2026-11-01

- Quartus target: DE10-Lite, MAX 10 `10M50DAF484C7G`, 50 MHz system clock.
- Fitter usage: 6,952 / 49,760 logic elements; 88 / 182 M9K blocks; 635,248 memory bits; 5 embedded multiplier elements.
- Slow 1200 mV, 85 °C timing model: Fmax 59.11 MHz and setup slack +3.081 ns at the 50 MHz clock constraint.
- The compiled `.sof` was programmed to the board successfully. The ESP32-S3 firmware was also compiled and flashed successfully.
- One physical UART transaction using the `FPGA TEST` command completed and passed response framing/CRC/ID checks. The FPGA returned class `S`, 57.66% calibrated top-1 probability, 872,541 cycles (17.451 ms at 50 MHz), and rejected the glyph. UART round trip was 22.445 ms. This generated test glyph verifies the link and inference response path; it is not an accuracy measurement.
- An OV3660 frame was captured successfully, but the captured image was too dark and did not show a readable plate. A complete live-camera plate recognition result and its accuracy have therefore not yet been verified.
- The independent kit flow is: OV3660 capture and glyph preprocessing on ESP32-S3 → eight sequential 28×28 grayscale UART requests → LeNet inference, confusion-posterior calibration, top-3 ranking, and character/plate accept-or-reject decisions on DE10-Lite → FPGA results returned to ESP32-S3 → result and per-character diagnostics shown in the web UI.
