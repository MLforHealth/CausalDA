# Pulling Up by the Causal Bootstraps: Causal Data Augmentation for Pre-training Debiasing

## Paper
If you use this code in your research, please cite the following publication:
```
TODO
```

This paper can also be found on arxiv: TODO


## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:MLforHealth/CausalDA.git
cd CausalDA/
conda env create -f environment.yml
conda activate causalda
```

### Step 1: Obtaining the Data
See [DataSources.md](DataSources.md) for detailed instructions to setup the WILDS and CXR datasets. This is not necessary for the synthetic experiments.

### Step 2: Running Experiments

To train a single model, e.g.
```
python train_synthetic.py \
    --type par_back_front \
    --corr-coff 0.75 \
    --test-corr 0.75 \
    --output_dir /path/to/output
```

or 

```
python train.py \
    --type back \
    --data camelyon \
    --data_type Conf \
    --domains 2 3 \
    --corr-coff 0.95 \
    --seed 0 \
    --output_dir /path/to/output
```

To reproduce the experiments in the paper by training grids of models, call `sweep.py` using the class names defined in `experiments.py` as experiment names, e.g.

```
python sweep.py launch \
    --experiment CXR \
    --output_dir /my/sweep/output/path \
    --command_launcher "local" 
```

This command can also be ran easily using `launch_scripts/launch_exp.sh`. You will likely need to update the launcher to fit your compute environment.


### Step 3: Aggregating Results

We provide sample code for creating aggregate results for an experiment in `AggResults.ipynb`.



## Acknowledgements

We make use of code from the [WILDS benchmark](https://github.com/p-lambda/wilds) as well as from the [DomainBed framework](https://github.com/facebookresearch/DomainBed).


## License
This source code is released under the MIT license, included [here](LICENSE).
