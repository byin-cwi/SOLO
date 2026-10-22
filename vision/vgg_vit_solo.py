"""SOLO / BP on CIFAR-10 with VGG-11 and ViT-Tiny (entry point; architectures live in solo_cifar.py).

  ARCH=vgg11: VGG-11 + BN, CIFAR variant (8 conv blocks + fc, ~9.2M params)
  ARCH=vit:   ViT-Tiny style (patch 4, dim 192, depth 8, 3 heads, ~5.4M; VIT_DIM/VIT_DEPTH/VIT_HEADS)

Usage:
  ARCH=vgg11 MODE=bp   python vgg_vit_solo.py
  ARCH=vgg11 MODE=solo GS_MODE=jacobi python vgg_vit_solo.py
  ARCH=vit   MODE=bp   LR=0.01 python vgg_vit_solo.py
  ARCH=vit   MODE=solo GS_MODE=jacobi LR=0.01 python vgg_vit_solo.py
Other env vars as solo_cifar.py. For ARCH=vit, LR defaults to 0.01 and WD to 5e-4.
"""

import os

if __name__ == '__main__':
    os.environ.setdefault('ARCH', 'vgg11')
    if os.environ['ARCH'] == 'vit':
        os.environ.setdefault('LR', '0.01')       # ViT with SGD diverges at large lr
        os.environ.setdefault('WD', '5e-4')
    import solo_cifar
    if os.environ.get('VERIFY'):
        solo_cifar.verify()
    else:
        solo_cifar.train_cifar10(
            n_epochs=int(os.environ.get('EPOCHS', '50')),
            batch_size=int(os.environ.get('BATCH', '128')),
            lr=float(os.environ.get('LR', '0.1')),
            K=int(os.environ.get('K_MODULES', '2')),
            gs_mode=os.environ.get('GS_MODE', 'gs'),
            mode=os.environ.get('MODE', 'solo'),
            sched=os.environ.get('SCHED', 's3'),
            lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
            a_tgt=float(os.environ.get('A_TGT', '0.15')),
            alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
            mu=float(os.environ.get('MU', '1.0')),
            ratchet=os.environ.get('RATCHET', '1') == '1',
            delta_t=float(os.environ.get('DELTA_T', '0.2')),
            aux_ce=float(os.environ.get('AUX_CE', '0.1')),
            lam=float(os.environ.get('LAM', '1.0')),
            align_type=os.environ.get('ALIGN_TYPE', 'none'),
            label_smoothing=(float(os.environ['LABEL_SMOOTH'])
                             if 'LABEL_SMOOTH' in os.environ else None),
            autoaug=os.environ.get('AUTOAUG', '1') == '1',
            carry=os.environ.get('CARRY', '0') == '1',
            data_root=os.environ.get('DATA_ROOT', './data'),
        )
