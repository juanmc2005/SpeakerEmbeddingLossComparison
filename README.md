# A Comparison of Metric Learning Loss Functions for End-to-End Speaker Verification

This is the companion repository for the paper [A Comparison of Metric Learning Loss Functions for End-to-End Speaker Verification](https://arxiv.org/abs/2003.14021). It hosts our best model trained with additive angular margin loss, and contains instructions for reproducing our results and using the model.

The project depends on the [pyannote-audio](https://github.com/pyannote/pyannote-audio) toolkit, so make sure you install it before running any code.

## Architecture

![Architecture of the model](images/architecture.png?raw=true "Architecture")

The architecture of our model consists of [SincNet](https://arxiv.org/abs/1808.00158) for feature extraction followed by [x-vector](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf).

## Training

You can train our model from scratch using the configuration file `config.yml` that we provide. All you need to do is run the following commands in your terminal:

```console
$ export EXP=models/AAM # Replace with the new path to config.yml
$ export PROTOCOL=VoxCeleb.SpeakerVerification.VoxCeleb2
$ pyannote-audio emb train --parallel=10 --gpu --to=1000 $EXP $PROTOCOL 
```

Note that you may need to change parameters based on your setup.

## Evaluation

We provide a step-by-step guide on reproducing our equal error rates alongside their 95% confidence intervals. The guide first evaluates the pretrained model using raw cosine distances, and then improves it with adaptive s-norm score normalization.

If you want to reproduce our results, check out [this notebook](https://github.com/juanmc2005/SpeakerEmbeddingLossComparison/blob/master/reproduce.ipynb)

## Fine-tuning

You can fine-tune our model to your dataset with the following commands:

```console
$ export WEIGHTS=models/AAM/train/VoxCeleb.SpeakerVerification.VoxCeleb2.train/weights/0560.pt
$ export EXP=<your_experiment_directory>
$ export PROTOCOL=<your_pyannote_database_protocol>
$ pyannote-audio emb train --pretrained $WEIGHTS --gpu --to=1000 $EXP $PROTOCOL
```

## Usage in Python

The default pyannote model for speaker embedding on `torch.hub` is our AAM loss model trained on variable length audio chunks. If you want to use the model right away, you can do so easily in a Python script:

```python
# load pretrained model from torch.hub
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'emb')

# extract embeddings for the whole files
emb1 = model({'audio': '/path/to/file1.wav'})
emb2 = model({'audio': '/path/to/file2.wav'})

# compute distance between embeddings
from scipy.spatial.distance import cdist
import numpy as np
distance = np.mean(cdist(emb1, emb2, metric='cosine'))
```

You can also replace the call to `torch.hub.load` with a pyannote `Pretrained` instance pointing to the model in this repo.
Note that this also allows for customized fine-tuning as well.

## Citation

If our work has been useful to you, please cite our paper:

```bibtex
@misc{coria2020comparison,
    title={{A Comparison of Metric Learning Loss Functions for End-To-End Speaker Verification}},
    author={Juan M. Coria and Hervé Bredin and Sahar Ghannay and Sophie Rosset},
    year={2020},
    eprint={2003.14021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
