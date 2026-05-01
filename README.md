# EEG Foundation Models

## Pretrained EEG encoders we depend on


| Model   | arXiv                                          | GitHub                                                                          | Open Source Model                                                           |
| ------- | ---------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| REVE    | [2510.21585](https://arxiv.org/abs/2510.21585) | [elouayas/reve_eeg](https://github.com/elouayas/reve_eeg)                       | [brain-bzh/reve](https://huggingface.co/collections/brain-bzh/reve)         |
| DIVER-1 | [2512.19097](https://arxiv.org/abs/2512.19097) | [DIVER-1](https://anonymous.4open.science/r/DIVER-1/README.md)                  | [DIVER-1](https://anonymous.4open.science/r/DIVER-1/README.md)              |
| TFM     | [2502.16060](https://arxiv.org/abs/2502.16060) | [Jathurshan0330/TFM-Tokenizer](https://github.com/Jathurshan0330/TFM-Tokenizer) | [Jathurshan/TFM-Tokenizer](https://huggingface.co/Jathurshan/TFM-Tokenizer) |


## Experiments


| #     | Track                                                                 | Status                                                                                                 |
| ----- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| exp01 | [EEG → text with frozen-LM bridge](experiments/exp01_eeg_to_text/)    | Closed; §4.3 negative; pivot logged in `next_experiments.md`                                           |
| exp02 | [EEG → text with CTC head](experiments/exp02_eeg_ctc/)                | Wave-3 in flight; see `progress.md` / `findings.md`                                                    |
| exp03 | [Self-supervised EEG pretraining](experiments/exp03_eeg_pretraining/) | Insights/methodology playbook in `[methodology.md](experiments/exp03_eeg_pretraining/methodology.md)`  |


