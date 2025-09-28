chmod +x docker_src/docker_test.py
chmod +x docker_src/docker_train.py

# python docker_src/docker_test.py \
python docker_src/docker_train.py \
    --dataset_name "ylecun/mnist"