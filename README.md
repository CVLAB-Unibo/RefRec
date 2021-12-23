# RefRec
Official repository for "RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation"

[[Project page]](https://cvlab-unibo.github.io/RefRec/) [[Paper]](https://arxiv.org/abs/2110.11036)

### Authors

[Adriano Cardace](https://www.unibo.it/sitoweb/adriano.cardace2) - [Riccardo Spezialetti](https://www.unibo.it/sitoweb/riccardo.spezialetti) - [Pierluigi Zama Ramirez](https://pierlui92.github.io/) - [Samuele Salti](https://vision.deis.unibo.it/ssalti/) - [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano/)


## Requirements
We rely on several libraries: [Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning), [Weight & Biases](https://docs.wandb.ai/), [Hesiod](https://github.com/lykius/hesiod)

To run the code, please follow the instructions below.

1) install required dependencies

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
2) Install pytorchEMD following https://github.com/daerduoCarey/PyTorchEMD and https://github.com/daerduoCarey/PyTorchEMD/issues/6 for latest versions of Torch


## Download and load datasets on W&B server (Registration required)
Reqeuest dataset access at https://drive.google.com/file/d/14mNtQTPA-b9_qzfHiadUIc_RWvPxfGX_/view?usp=sharing
The dataset is the same provided by the original authours at https://github.com/canqin001/PointDAN. For convenience we provide a preprocessed version used in this work.

To train the reconstruction network merge two datasets. Then, load all the required datasets on wandb server. For example for modelnet->scannet execute the following commands:

```bash
unzip PointDA_aligned.zip -d data/
cd data
cp modelent modelnet_scannet 
rsync -av scannet modelnet_scannet
./load_data.sh
```

Then you can simply train the whole pipeline with 
```bash
./train_pipeline_m2sc.sh
```

