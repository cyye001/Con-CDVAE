# Con-CDVAE

此代码在 [CDVAE](https://arxiv.org/abs/2110.06197) 的基础上改进，实现了根据目标属性生成晶体。

参考文献：[Cai-Yuan Ye, Hong-Ming Weng, Quan-Sheng Wu, Con-CDVAE: A method for the conditional generation of crystal structures, Computational Materials Today, 1, 100003 (2024).](https://www.sciencedirect.com/science/article/pii/S2950463524000036)

arXiv: [https://arxiv.org/abs/2403.12478](https://arxiv.org/abs/2403.12478)

## 更新
- v2.1.0 更新了先验模块和结构生成的代码
- v2.0.0 使用新的PyTorch环境
- v1.0.0 Con-CDVAE的初始实现

> **提示：** 2.x版本目前正在积极开发中，可能不稳定。

## 环境

我们推荐使用Anaconda管理Python环境。首先，创建并激活一个新的Python环境：

```
conda create --name concdvae310 python=3.10
conda activate concdvae310
```

然后，使用`requirements.txt`安装Python包。

```
pip install -r requirements.txt
```

最后，根据你的设备和CUDA版本安装PyTorch相关库。我们使用的版本是：

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
详情请参考 [PyTorch](https://pytorch.org), [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/#), [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/).

环境设置完成后，你可以使用提供的模型检查点运行Con-CDVAE，以条件生成材料。在运行之前，请确保更新必要的环境路径。你可以运行以下命令：

```
cp .env_bak .env
bash writeenv.sh
```

或者，你也可以手动修改.env文件。将其更新为以下内容，将<YOUR_PATH_TO_CONCDVAE>替换为你的Con-CDVAE目录的绝对路径：

```
export PROJECT_ROOT="<YOUR_PATH_TO_CONCDVAE>"
export HYDRA_JOBS="<YOUR_PATH_TO_CONCDVAE>/output/hydra"
export WABDB_DIR="<YOUR_PATH_TO_CONCDVAE>/output/wandb"
```

## 数据集

你可以在`data/`中找到数据集的小样本（`mptest/`和`mptest4conz`），包括用于Con-CDVAE两步训练的数据。完整数据可以根据[Materials Project (MP)](https://next-gen.materialsproject.org/)和[Open Quantum Materials Database (OQMD)](https://oqmd.org/)提供的API轻松下载，并且它们可以使用与样本相同的格式。

## 使用预训练模型
一个预训练模型可在`src/model/mp20_format`中找到，该模型在mp_20数据集上训练。它可以根据形成能生成晶体结构。此模型可能与论文中呈现的结果不完全匹配，因为它是使用修改后的代码重新训练的。

使用以下命令，通过`default`策略生成晶体：

```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/default.yaml
```

使用以下命令，通过`full`策略生成晶体：

```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/full.yaml
```

使用以下命令，通过`less`策略生成晶体：

```
python scripts/gen_crystal.py --config <YOUR_PATH_TO_CONCDVAE>/conf/gen/less.yaml
```

控制生成参数的配置文件位于`conf/gen/`。你可以参考`src/model/mp20_format`中的两个CSV文件作为模型输入。

晶体结构生成后，它们将保存在与模型相同的目录下，文件名如`eval_gen_xxx.pt`，其中xxx对应于你的YAML和CSV文件中指定的设置。

## 训练Con-CDVAE

### 第一步训练
要训练Con-CDVAE，首先运行以下命令：

```
python concdvae/run.py data=mptest expname=test model=vae_mp_format
```

要使用其他数据集，用户应以与样本相同的格式准备数据，并在`conf/data/`文件夹中编辑一个新的配置文件，然后使用`data=your_data_conf`。要训练其他属性的模型，可以尝试`model=vae_mp_gap`。

如果你想使用多个GPU加速，应运行以下命令：

```
torchrun --nproc_per_node 4 concdvae/run.py \
    data=mptest \
    expname=test \
    model=vae_mp_gap \
    train.pl_trainer.accelerator=gpu  \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true 
```
训练后，模型检查点可以在`<YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname>/epoch=xxx-step=xxx.ckpt`中找到。

### 第二步训练
完成第一步训练后，你可以使用以下命令训练*先验*模块：

```
python concdvae/run_prior.py \
  --model_path <YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname> \
  --model_file epoch=xxx-step=xxx.ckpt
  --prior_label prior_default
```
然后你可以在`<YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname>/prior_default-epoch=xxx-step=xxx.ckpt`中得到默认条件的*先验*。

如果你想训练全条件的*先验*，你应该使用：

```
python concdvae/run_prior.py \
  --model_path <YOUR_PATH_TO_CONCDVAE>/output/hydra/singlerun/YYYY-MM-DD/<expname> \
  --prior_label prior_full \
  --priorcondition_file mp_full \
  --data_file mptest4conz
```

## 评估模型

要评估晶体系统，你可以使用代码`concdvae/pt2CS.py`。

要评估其他属性，你应该训练一个[CGCNN](https://github.com/txie-93/cgcnn)，使用以下命令：

```
python cgcnn/main.py /your_path_to_con-cdvae/cgcnn/data/mptest --prop band_gap --label your_label 
```
此代码使用与Con-CDVAE相同的数据集，你可以使用前面提到的方法构建所需的数据库。如果你想训练其他属性的CGCNN，可以设置`--prop formation_energy_per_atom`、`--prop BG_type`、`--prop FM_type`。需要注意的是，如果你正在训练分类任务，你应该设置`--task classification`。

训练后，模型检查点可以在`your_labelmodel_best.pth.tar`中找到。训练好的模型可以在`cgcnn/pre-trained`中找到。

当你生成晶体并需要评估时，运行以下命令：

```
python cgcnn/predict.py --gendatapath /your_path_to_generated_crystal/ --modelpath /your_path_to_cgcnn_model/model_best.pth.tar --file your_crystal_file.pt --label your_label
```

[English README](README.md)

```
