# A Comparison of Metric Learning Loss Functions for End-to-End Speaker Verification

This is the companion repository of the paper "A Comparison of Metric Learning Loss Functions for End-to-End Speaker Verification". It contains our best model trained with additive angular margin loss.

This model is dependant on the [pyannote-audio toolkit](https://github.com/pyannote/pyannote-audio), so make sure you install it if you plan to use our pretrained model.

## Training

You can train our model from scratch using the configuration file `config.yml` that we provide. All you need to do is run the following command in your terminal:

```console
$ export EXP=models/AAM # Replace with the new path to config.yml
$ export PROTOCOL=VoxCeleb.SpeakerVerification.VoxCeleb2
$ pyannote-audio emb train --parallel=10 --gpu --to=1000 $EXP $PROTOCOL 
```

Note that you may need to change parameters based on your setup.

## Evaluation

We provide a step-by-step guide on reproducing our equal error rates alongside their 95% confidence intervals. The guide first evaluates the pretrained model using raw cosine distances, and then applying adaptive s-norm score normalization.

If you want to reproduce our results, check out [this notebook](https://github.com/juanmc2005/SpeakerEmbeddingLossComparison/blob/master/reproduce.ipynb)

## Fine-tuning

You can fine-tune our model to your dataset with the following command:

```console
$ export WEIGHTS=models/AAM/train/VoxCeleb.SpeakerVerification.VoxCeleb2.train/weights/0560.pt
$ export EXP=<your_experiment_directory>
$ export PROTOCOL=<your_pyannote_database_protocol>
$ pyannote-audio emb train --pretrained $WEIGHTS --subset=train --gpu --to=1000 $EXP $PROTOCOL
```

## Citation

If you use this in your work, please cite our paper:

```bibtex
BibTeX coming soon!
```
