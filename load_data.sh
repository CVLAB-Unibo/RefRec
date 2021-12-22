entity="3dv"
project="s2sc"

python utils/load_dataset.py shapenet_scannet $entity $project
python utils/load_dataset.py shapenet $entity $project
python utils/load_dataset.py scannet $entity $project
