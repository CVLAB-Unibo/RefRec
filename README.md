# RefRec
Official repository for "RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation"

[[Project page]](https://cvlab-unibo.github.io/RefRec/) [[Paper]](https://arxiv.org/abs/2110.11036)

- [Adriano Cardace](https://www.unibo.it/sitoweb/adriano.cardace2) - [Riccardo Spezialetti](https://www.unibo.it/sitoweb/riccardo.spezialetti) - [Pierluigi Zama Ramirez](https://pierlui92.github.io/) - [Samuele Salti](https://vision.deis.unibo.it/ssalti/) - [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano/)

Sill work in progress........

To run the code please follow the instructions below.

python -m venv env
pip install -r requirements.txt
source env/bin/activate
install pytorchEMD following https://github.com/daerduoCarey/PyTorchEMD

# Load datasets on W&B server
unzip data/
./load_data.sh
