# DE10-Lite hardware bring-up

This folder contains the first verified Quartus project for the board. It is a
clock/LED heartbeat only: it proves that the DE10-Lite `10M50DAF484C7G` target,
pin assignments, Quartus compilation and USB-Blaster programming path work.

It is **not** the LeNet-5 OCR accelerator yet. The current repository has the
trained PyTorch model and the software pipeline, but no synthesizable RTL for
perspective correction, row sorting, character buffering, fixed-point LeNet-5,
or image input. Those blocks must be implemented and validated before an OCR
bitstream can honestly be claimed.

Compile from the repository root:

```powershell
E:\k\quartus\bin64\quartus_sh.exe --flow compile hardware\de10_lite_bringup\Bien_so.qpf
```

Program the resulting bitstream when the board is connected:

```powershell
E:\k\quartus\bin64\quartus_pgm.exe -c 1 -m jtag -o "p;hardware\de10_lite_bringup\output_files\Bien_so.sof"
```
