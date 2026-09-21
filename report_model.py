from pathlib import Path

import torch

from structured_prune_lenet5 import LeNet5Structured


CHECKPOINT = Path('artifacts/lenet5_ocr_structured_pruned25_final.pt')


def main():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = LeNet5Structured(36)
    model.load_state_dict(checkpoint['model'])
    params = sum(parameter.numel() for parameter in model.parameters())
    int8_mib = params / (1024 * 1024)
    mac_char = model.macs()
    print('\n===== BAO CAO MO HINH STRUCTURED LENET-5 =====')
    print(f'Checkpoint          : {CHECKPOINT}')
    print(f'Channels            : {checkpoint.get("channels", model.channels)}')
    print(f'So tham so          : {params:,}')
    print(f'Trong so INT8       : {int8_mib:.3f} MiB')
    print(f'MAC mot ky tu       : {mac_char:,}')
    print(f'MAC bien 1 hang (8) : {mac_char * 8:,}  (~{mac_char * 8 / 1e6:.2f} trieu)')
    print(f'MAC bien 2 hang (16): {mac_char * 16:,} (~{mac_char * 16 / 1e6:.2f} trieu)')
    print(f'Accuracy validation : {checkpoint.get("val_acc", 0.0) * 100:.2f}%')
    print('===============================================\n')


if __name__ == '__main__':
    main()
