```
pip3 install -U scikit-learn

pip3 install xgboost

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install gbnet

pip3 install tensorflow[and-cuda] # This has some conflicts
pip3 install tensorflow_decision_forests
# Note: https://www.tensorflow.org/decision_forests/migration#:~:text=g.-,GPU%2C%20TPU,per%20example%20per%20cpu%20core%29.
# TF-DF training does not (yet) support hardware accelerators. All training and inference is done on the CPU (sometimes using SIMD).

pip3 install -U "jax[cuda12]" # This has a conflict
pip3 install git+https://github.com/AmedeoBiolatti/jaxgboost

##################################################
# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH=$PATH:~/miniconda3/bin
conda init

# restart the terminal
conda env create --name xgb-gpu118 --file /home/ubuntu/visamlp/gpu-expr/envs/xgboost-env/environment.yml

git config --global user.email "adam.yufeng.hsing@gmail.com"
git config --global user.name "ufenx"

# dask version
conda env create --name xgb-gpu118-dask --file /home/ubuntu/visamlp/gpu-expr/envs/xgboost-env/dask-env.yml
python -m ipykernel install --user --name xgb-gpu118-dask --display-name "xgb-gpu118-dask"
conda install -c conda-forge ucx ucx-proc=*=gpu
conda install -c conda-forge dask-cuda

conda env create --name jaxgboost --file /home/ubuntu/visamlp/gpu-expr/envs/jax-xgboost/jaxgboost.yml
python -m ipykernel install --user --name jaxgboost --display-name "jaxgboost"

#################################

conda create -n xgboost-env
conda activate xgboost-env
conda install xgboost
conda install scikit-learn
conda install ipykernel
python -m ipykernel install --user --name=xgboost-env
cd ../envs/xgboost-env/
pip list --format=freeze > requirements.txt

conda uninstall xgboost
export CONDA_OVERRIDE_CUDA="12.8"
conda install -c conda-forge py-xgboost=*=cuda*
conda uninstall py-xgboost
conda install -c conda-forge py-xgboost-gpu
# not work ####

### working one ###
conda create -n xgb-gpu118 python=3.10
conda activate xgb-gpu118
conda install -c conda-forge py-xgboost-gpu cudatoolkit=11.8
conda install scikit-learn
conda install ipykernel
python -m ipykernel install --user --name=xgb-gpu118
conda list -e > requirements.txt # this is not working
conda env export > environment.yml

conda deactivate
conda remove --name xgb-gpu118 --all

rm -rf `find . -type d -name ".ipynb_checkpoints"`
```