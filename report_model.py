from pathlib import Path
import argparse

import torch

from structured_prune_lenet5 import LeNet5Structured
from train_lenet5 import LeNet5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/lenet5_ocr_structured_pruned25_final.pt'))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = LeNet5Structured(36) if checkpoint.get('arch') == 'LeNet5Structured' else LeNet5(36)
    model.load_state_dict(checkpoint['model'])
    params = sum(parameter.numel() for parameter in model.parameters())
    int8_mib = params / (1024 * 1024)
    if hasattr(model, 'macs'):
        dense_mac_char = model.macs()
        mac_layers = None
    else:
        mac_layers = [('conv1', 117600, model.features[0]), ('conv2', 240000, model.features[3]),
                      ('fc1', 48000, model.classifier[0]), ('fc2', 10080, model.classifier[2]),
                      ('fc3', 3024, model.classifier[4])]
        dense_mac_char = sum(mac for _, mac, _ in mac_layers)
    effective_mac_char = dense_mac_char
    if mac_layers:
        effective_mac_char = sum(mac * int((layer.weight != 0).sum()) / layer.weight.numel()
                                  for _, mac, layer in mac_layers)
    weight_total = sum(layer.weight.numel() for _, _, layer in mac_layers) if mac_layers else None
    weight_nonzero = sum(int((layer.weight != 0).sum()) for _, _, layer in mac_layers) if mac_layers else None
    print('\n===== BAO CAO MO HINH LENET-5 =====')
    print(f'Checkpoint          : {args.checkpoint}')
    print(f'Channels            : {checkpoint.get("channels", "6,16,120,84 (dense shape)")}')
    print(f'So tham so          : {params:,}')
    print(f'Trong so INT8       : {int8_mib:.3f} MiB')
    print(f'MAC day du mot ky tu: {dense_mac_char:,}')
    print(f'MAC hieu dung bo qua weight=0: {effective_mac_char:,.0f}')
    print(f'MAC bien 1 hang (8 ky tu): {effective_mac_char * 8:,.0f} (~{effective_mac_char * 8 / 1e6:.2f} trieu)')
    print(f'MAC bien 2 hang (8 ky tu): {effective_mac_char * 8:,.0f} (~{effective_mac_char * 8 / 1e6:.2f} trieu)')
    if weight_total:
        print(f'Weight non-zero       : {weight_nonzero:,}/{weight_total:,} ({weight_nonzero / weight_total * 100:.2f}%)')
    print(f'Accuracy validation : {checkpoint.get("val_acc", 0.0) * 100:.2f}%')
    print('===============================================\n')


if __name__ == '__main__':
    main()
