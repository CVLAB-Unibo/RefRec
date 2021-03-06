entity="3dv"
dataset_source="modelnet"
dataset_target="scannet"
prototypes_path="prototypes/m2sc.npy"
hard_split="m2sc_hard_split"
easy_split="m2sc_easy_split"

python main_reconstruction.py logs/m_2_sc_rec/run.yaml
python main.py logs/warmup_m2sc/run.yaml

python utils/generate_pseudo_labels.py  \
                                    --entity=$entity \
                                    --dataset_source=$dataset_source \
                                    --dataset_target=$dataset_target \
                                    --restore_weights=warmup_m2sc \
                                    --restore_weights_rec=m_2_sc_rec \
                                    --easy_split=$easy_split \
                                    --hard_split=$hard_split \
                                    --prototypes_path=$prototypes_path

python main.py logs/self_train_m2sc/run.yaml \
                                    --prototypes_path=$prototypes_path \
                                    --dataset_source=$easy_split \
                                    --dataset_hard=$hard_split 