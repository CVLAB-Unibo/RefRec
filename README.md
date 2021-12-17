# RefRec
Official repository for "RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation"

python -m venv env
pip install -r requirements.txt
source env/bin/activate
install pytorchEMD following https://github.com/daerduoCarey/PyTorchEMD

# Load datasets on W&B server
unzip data/
./load_data.sh