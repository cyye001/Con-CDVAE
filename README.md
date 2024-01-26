# Con-CDVAE

This code is improved on the basis of 
[CDVAE](https://arxiv.org/abs/2110.06197), 
and implements the generation of crystals according to 
the target properties.

## Installation
It easy to building a python environment using conda.
Run the following command to install the environment:
```bash
conda env create -f environment.yml
```

Modify the following environment variables in `.env`.

- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs

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
python concdvae/run.py train=new data=mptest expname=test model=vae_mp_CSclass
```

To use other dataset, user should prepare the data in the same forme as 
the sample, and edit a new configure files in `conf/data/` folder, 
and use `data=your_data_conf`. To train model for other property, use 
`model=vae_mp_format` or `model=vae_mp_gap`. 

If you want to accelerate with a gpu, you should set `accelerator=gpu`
in command line. If you want to accelerate with multiple gpus, you should
run this command:
```
torchrun --nproc_per_node 4 concdvae/run.py train=new data=mptest expname=test model=vae_mp_CSclass accelerator=ddp
```
After training, model checkpoints can be found in
`$HYDRA_JOBS/singlerun/YYYY-MM-DD/model_expname.pth`.


### Step-two training
After finishing step-one training, you can train the *Prior* block
with the following command.
```
python scripts/condition_diff_z.py --model_path /your_path_to_model_checkpoints/ --model_file model_expname.pth --fullfea 0 --label your_label
```
Then you can get the default condition *Prior* in 
`/your_path_to_model_checkpoints/conz_model_your_label_diffu.pth`.

If you want to train full conditon *Prior*, you should change 
`--fullfea 0` to `--fullfea 1` and set
`--newcond /your_path_to_conf/conf/conz_2.yaml --newdata mptest4conz`

## Generating crystals with target propertise
To generate materials, you should prepare condition file. 
You can see the example in `/output/hydra/singlerun/2024-01-25/test/`,
where "general_full.csv" is for *default* strategy or *full* strategy, 
and "general_less.csv" is for *less* strategy.

Then run the following command:
```
python scripts/evaluate_diff.py --model_path  --model_file/your_path_to_model_checkpoints/ model_expname.pth  --conz_file conz_model_your_label_diffu.pth  --label your_label --prop_path general_full.csv
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
