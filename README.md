# CLCFI

The official implementation of CLCFI: Multimodal Sentiment Analysis via Cue-Level Counterfactual Intervention.

## Datasets

The experiments are conducted on CMU-MOSI, CMU-MOSEI,and CH-SIMS.

## Training

```bash
python -m CLCFI.train --dataset mosi
python -m CLCFI.train --dataset mosei
python -m CLCFI.train --dataset sims
```

## Environment Requirements

Python == 3.10

PyTorch == 2.4.1

CUDA == 12.1
