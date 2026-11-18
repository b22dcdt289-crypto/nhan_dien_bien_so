# ESP32-S3 & DE10-Lite license plate recognition

ESP32-S3/OV3660 preprocessing and UART interface, LeNet K3 FPGA RTL, and synthetic-data experiments.

- `esp32/`: existing one-row camera firmware.
- `fpga/standard/`: existing 30-class UART RTL and weights.
- `fpga/experimental/`: separately tested parallel RTL and buffered high-speed UART; requires compatible ESP32 firmware and hardware validation.
- `training/` and `model/`: experimental 31-class multirow model and scripts; **not compatible with the existing 30-class FPGA bitstream**.
- `docs/`: experimental results, protocol and limitations.

The Python research scripts currently rely on additional files and datasets from `D:/TOTNGHIEP/license_plate_ai` and local synthetic templates/fonts; this export does not yet include all dependencies needed to reproduce training. The ESP32 remains limited to one-row eight-character plates. Metrics are from synthetic images, not verified real-world accuracy.

To avoid publishing real plate identities, the exclusion list in `training/build_data.py` is empty in this export. Supply your own held-out identities before generating data and evaluating the model; the export alone cannot reproduce the original split.

No camera captures, credentials, build output or personal documents are included.
