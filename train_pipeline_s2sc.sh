entity="3dv"
dataset_source="shapenet"
dataset_target="scannet"
prototypes_path="prototypes/s2sc.npy"
hard_split="s2sc_hard_split"
easy_split="s2sc_easy_split"

python main_reconstruction.py logs/s_2_sc_rec/run.yaml
python main.py logs/warmup_s2sc/run.yaml

python utils/generate_pseudo_labels.py  \
                                    --entity=$entity \
                                    --dataset_source=$dataset_source \
                                    --dataset_target=$dataset_target \
                                    --restore_weights=warmup_s2sc \
                                    --easy_split=$easy_split \
                                    --hard_split=$hard_split \
                                    --prototypes_path=$prototypes_path

python main.py logs/self_train_s2sc/run.yaml \
                                    --prototypes_path=$prototypes_path
                                    --dataset_source=$easy_split
                                    --dataset_hard=$hard_split

python main.py logs/self_train_s2sc/run.yaml \
                                    --prototypes_path=$prototypes_path
                                    --dataset_source=$easy_split
                                    --dataset_hard=$hard_split

python main.py logs/self_train_s2sc/run.yaml \
                                    --prototypes_path=$prototypes_path
                                    --dataset_source=$easy_split
                                    --dataset_hard=$hard_split

python main.py logs/self_train_s2sc/run.yaml \
                                    --prototypes_path=$prototypes_path
                                    --dataset_source=$easy_split
                                    --dataset_hard=$hard_split