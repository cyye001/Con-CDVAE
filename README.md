# Con-CDVAE

This code is improved on the basis of 
[CDVAE](https://arxiv.org/abs/2110.06197), 
and implements the generation of crystals according to 
the target properties.

Ref: [Cai-Yuan Ye, Hong-Ming Weng, Quan-Sheng Wu, Con-CDVAE: A method for the conditional generation of crystal structures, Computational Materials Today, 1, 100003 (2024).](https://www.sciencedirect.com/science/article/pii/S2950463524000036)

arXiv: [https://arxiv.org/abs/2403.12478](https://arxiv.org/abs/2403.12478)


## Update
- v2.1.0 Updated the code for the prior module and the structure generation
- v2.0.0 Use the new PyTorch environment
- v1.0.0 Initial implementations of Con-CDVAE

> **Tip:** Version 2.x is currently under active development and may be unstable.

## Environment

We recommend using Anaconda to manage Python environments. First, create and activate a new Python environment:
```
conda create --name concdvae310 python=3.10
conda activate concdvae310
```

Then, use `requirements.txt` to install the Python packages.
```
pip install -r requirements.txt
```

Finally, the PyTorch-related libraries need to be installed according to your device and CUDA version. The version we used is:
```
torch                    2.3.0+cu118
torchaudio               2.3.0+cu118
torchvision              0.18.0+cu118

torch_geometric          2.5.3
torch_cluster            1.6.3+pt23cu118
torch_scatter            2.1.2+pt23cu118
torch_sparse             0.6.18+pt23cu118
torch_spline_conv        1.2.2+pt23cu118

pytorch-lightning        2.4.0
torchmetrics             1.6.3
```
For details, you can refer to [PyTorch](https://pytorch.org), [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/#), [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/).


After setting up the environment, you can use the provided model checkpoint to run PODGen for conditional generation of topological materials. Before doing so, make sure to update the necessary environment paths. You can either run the following commands:

```
cp .env_bak .env
bash writeenv.sh
```

Or, if you prefer, modify the .env file manually. Update it with the following lines, replacing <YOUR_PATH_TO_CONCDVAE> with the absolute path to your PODGen directory:


```
export PROJECT_ROOT="<YOUR_PATH_TO_CONCDVAE>"
export HYDRA_JOBS="<YOUR_PATH_TO_CONCDVAE>/output/hydra"
export WABDB_DIR="<YOUR_PATH_TO_CONCDVAE>/output/wandb"
```

## Datasets

You can find a small sample of the dataset in `data/` (`mptest/` and `mptest4conz` ), 
including the data used for Con-CDVAE two-step training. 
The complete data can be easily downloaded according to the API 
provided by the [Materials Project (MP)](https://next-gen.materialsproject.org/)
and [Open Quantum Materials Database (OQMD)](https://oqmd.org/),
and they can be used in the same format as the sample.

## Use the pre-train model
A pre-trained model is available in `src/model/mp20_format`, trained on the mp_20 dataset. It can generate crystal structures based on formation energy. This model may not exactly match the results presented in the paper, as it was retrained using the modified code.

Use the following command to generate crystals using the `default` strategy:
```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/default.yaml
```

Use the following command to generate crystals using the `full` strategy:
```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/full.yaml
```

Use the following command to generate crystals using the `less` strategy:
```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/less.yaml
```

The configuration files for controlling the generation parameters are located in `conf/gen/`. You can refer to the two CSV files in `src/model/mp20_format` for the model input.

After crystal structures are generated, they are saved in the same directory as the model under filenames like `eval_gen_xxx.pt`, where xxx corresponds to the settings specified in your YAML and CSV files.

## Training Con-CDVAE

### Step-one training
To train a Con-CDVAE, run the following command first:

```
python concdvae/run.py data=mptest expname=test model=vae_mp_format
```

To use other dataset, user should prepare the data in the same forme as 
the sample, and edit a new configure files in `conf/data/` folder, 
and use `data=your_data_conf`. To train model for other property, you can try
`model=vae_mp_gap`. 

If you want to accelerate with multiple gpus, you should
run this command:
```
torchrun --nproc_per_node 4 concdvae/run.py \
    data=mptest \
    expname=test \
    model=vae_mp_gap \
    train.pl_trainer.accelerator=gpu  \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true 
```
After training, model checkpoints can be found in
`<YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname>/epoch=xxx-step=xxx.ckpt`.


### Step-two training
After finishing step-one training, you can train the *Prior* block
with the following command.
```
python concdvae/run_prior.py \
  --model_path <YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname> \
  --model_file epoch=xxx-step=xxx.ckpt
  --prior_label prior_default
```
Then you can get the default condition *Prior* in 
`<YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname>/prior_default-epoch=xxx-step=xxx.ckpt`.

If you want to train full conditon *Prior*, you should use:
```
python concdvae/run_prior.py \
  --model_path <YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname> \
  --prior_label prior_full \
  --priorcondition_file mp_full \
  --data_file mptest4conz
```

## Evaluating model

To evaluate crystal system, you can use the code `concdvae/pt2CS.py`.

To evaluate other properties, you should train a 
[CGCNN](https://github.com/txie-93/cgcnn) with the following command:
```
python cgcnn/main.py /your_path_to_con-cdvae/cgcnn/data/mptest --prop band_gap --label your_label 
```
This code use the same dataset as Con-CDVAE, You can build 
the required database using the methods mentioned earlier.
If you want to train CGCNN on other property, you can set 
`--prop formation_energy_per_atom`, `--prop BG_type`, `--prop FM_type`.
It is important to note that if you are training for a 
classification task, you should set `--task classification`.

After training, model checkpoints can be found in
`your_labelmodel_best.pth.tar`. The trained model can be found in 
`cgcnn/pre-trained`.

When you've generated crystals and need to evaluate, 
run the following command:
```
python cgcnn/predict.py --gendatapath /your_path_to_generated_crystal/ --modelpath /your_path_to_cgcnn_model/model_best.pth.tar --file your_crystal_file.pt --label your_label
```
