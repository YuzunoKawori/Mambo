# Background Prompt for Few-Shot Out-of-Distribution Detection
## Requirements
The experiments were run on:
- Python 3.8.20 
- PyTorch 2.4.1
- Cuda 11.8
- This code is built on top of the awesome toolbox [Dassl.pytorch]
## Dataset
The overall file structure is as follows:
```
Mambo
|-- data
    |-- imagenet
        |-- imagenet-classes.txt
        |-- images/
            |--train/ # contains 1,000 folders like n01440764, n01443537, etc.
            |-- val/ # contains 1,000 folders like n01440764, n01443537, etc.
    |-- iNaturalist
    |-- SUN
    |-- Places
    |-- Texture
    ...
```
## Quick Start
The training script is in `Mambo/scripts/mambo/train.sh`

e.g., 16-shot training with ViT-B/16
```train
CUDA_VISIBLE_DEVICES=0 bash scripts/mambo/train.sh data imagenet vit_b16_ep50 end 16 16 False 0.2 200
```

The inference script is in `Mambo/scripts/mambo/eval.sh`.

Test ImageNet-1K OOD benchmark:
```eval
CUDA_VISIBLE_DEVICES=0 bash scripts/mambo/eval.sh data imagenet vit_b16_ep50 1 output/imagenet/Mambo/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1
