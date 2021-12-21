# RefRec
Official repository for "RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation"

[[Project page]](https://cvlab-unibo.github.io/RefRec/) [[Paper]](https://arxiv.org/abs/2110.11036)

### Authors

[Adriano Cardace](https://www.unibo.it/sitoweb/adriano.cardace2) - [Riccardo Spezialetti](https://www.unibo.it/sitoweb/riccardo.spezialetti) - [Pierluigi Zama Ramirez](https://pierlui92.github.io/) - [Samuele Salti](https://vision.deis.unibo.it/ssalti/) - [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano/)


## Requirements
We rely on several libraries: [Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Weight & Biases](https://docs.wandb.ai/), [Hesiod](https://github.com/lykius/hesiod)

To run the code, please follow the instructions below.

1) install pytorchEMD following https://github.com/daerduoCarey/PyTorchEMD

2) install required dependencies

```bash
python -m venv env
pip install -r requirements.txt
source env/bin/activate
```

## Dowdnload and load datasets on W&B server (Registration required)
unzip data/
./load_data.sh

Sill work in progress........
