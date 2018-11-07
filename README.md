## Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language Model (code dump)

This repository contains the code used for "Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language Model" (Sebastian J. Mielke, Jason Eisner, 2018).

This code is based on the [AWD-LSTM-LM](https://github.com/salesforce/awd-lstm-lm), updated independently to work well with PyTorch 0.4 and expanded with all sorts of things to allow for open-vocabulary LMing. These changes to pervade everything from data loading to batching to training to evaluation to output numbers, so diffing might prove to be tricky.

Sample invocations can be found in the accompanying shell files. Remnants of ideas are scattered throughout the code, fun gotchas (e.g., datasets that contain "-bpe" trigger the BPE baseline mode) come free of charge and many dead-looking codepaths are used to for example report the per-class numbers in Table 1 of the paper...

If you decide to use anything this code or the paper, please cite:
```
@article{MieEis18Spell,
  title={Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language Model},
  author={Mielke, Sebastian J. and Eisner, Jason},
  journal={arXiv preprint arXiv:1804.08205},
  year={2018}
}
```
or
```
@article{MieEis19Spell,
  title={Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary Language Model},
  author={Mielke, Sebastian J. and Eisner, Jason},
  journal={AAAI},
  year={2019}
}
```
