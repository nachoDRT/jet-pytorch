chmod +x docker_src/docker_test.py
chmod +x docker_src/docker_train.py

python docker_src/docker_test.py \
    --repo_id "de-Rodrigo/jet-mnist" \
    --model_name "jet_mnist"

# python docker_src/docker_train.py \
#     --dataset_name "ylecun/mnist" \
#     --wandb_entity "ciclab-comillas" \
#     --wandb_project "jet" \
#     --wandb_run_name "mnist" \
#     --hf_repo_id "de-Rodrigo/jet-mnist"