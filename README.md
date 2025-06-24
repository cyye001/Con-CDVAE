# Con-CDVAE

This code is improved on the basis of 
[CDVAE](https://arxiv.org/abs/2110.06197), 
and implements the generation of crystals according to 
the target properties.

Ref: [Cai-Yuan Ye, Hong-Ming Weng, Quan-Sheng Wu, Con-CDVAE: A method for the conditional generation of crystal structures, Computational Materials Today, 1, 100003 (2024).](https://www.sciencedirect.com/science/article/pii/S2950463524000036)

arXiv: [https://arxiv.org/abs/2403.12478](https://arxiv.org/abs/2403.12478)


## Update
- v2.0.0 Use the new PyTorch environment
- v1.0.0 Initial implementations of Con-CDVAE


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

Or, if you prefer, modify the .env file manually. Update it with the following lines, replacing <YOUR_PATH_TO_PODGEN> with the absolute path to your PODGen directory:


```
export PROJECT_ROOT="<YOUR_PATH_TO_PODGEN>/PODGen"
export HYDRA_JOBS="<YOUR_PATH_TO_PODGEN>/PODGen/output/hydra"
export WABDB_DIR="<YOUR_PATH_TO_PODGEN>/PODGen/output/wandb"
```

## Datasets

You can find a small sample of the dataset in `data/`, 
including the data used for Con-CDVAE two-step training. 
The complete data can be easily downloaded according to the API 
provided by the [Materials Project (MP)](https://next-gen.materialsproject.org/)
and [Open Quantum Materials Database (OQMD)](https://oqmd.org/),
and they can be used in the same format as the sample.

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
    model=vae_mp_format \
    train.pl_trainer.accelerator=gpu  \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true 
```
After training, model checkpoints can be found in
`$HYDRA_JOBS/singlerun/YYYY-MM-DD/<expname>/epoch=xxx-step=xxx.ckpt`.


### Step-two training
After finishing step-one training, you can train the *Prior* block
with the following command.
```
python scripts/condition_diff_z.py --model_path /your_path_to_model_checkpoints/ --model_file epoch=xxx-step=xxx.ckpt --fullfea 0 --label <your_label>
```
Then you can get the default condition *Prior* in 
`/your_path_to_model_checkpoints/conz_model_<your_label>_diffu.pth`.

<!-- If you want to train full conditon *Prior*, you should change 
`--fullfea 0` to `--fullfea 1` and set
`--newcond /your_path_to_conf/conf/conz_2.yaml --newdata mptest4conz`-->

## Generating crystals with target propertise
To generate materials, you should prepare condition file. 
You can see the example in `/src`,
where "general_full.csv" is for *default* strategy or *full* strategy, 
and "general_less.csv" is for *less* strategy.

Then run the following command:
```
python scripts/evaluate_diff.py --model_path /your_path_to_model_checkpoints/  --model_file model_expname.pth  --conz_file conz_model_your_label_diffu.pth  --label your_label --prop_path general_full.csv
```

If you want to filter latent variables using the *Predictor* block, set 
`--down_sample 100` which means filtering at a ratio of one hundred 
to one.

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
