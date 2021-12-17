entity="3dv"
project="m2sc"

python utils/load_dataset.py modelnet_scannet $entity $project
python utils/load_dataset.py modelnet $entity $project
python utils/load_dataset.py scannet $entity $project
