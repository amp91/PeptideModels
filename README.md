# Peptide Ligand Predictor 
______________________________

https://github.com/amp91/PeptideModels/assets/30113339/4126a37d-5260-4854-b5c7-481499ef6dec


## Introduction 
This repository hosts software that utilises machine learning models to design 
novel mono and dual agonists peptides targeting human GCG and GLP-1 receptors. 
It features a codebase designed to facilitate peptide sequence optimization, 
leveraging a pre-trained ensemble of deep multi-task neural network models to
tailor activity profiles according to specific requirements. Furthermore, the 
ensemble of convolutional multi-task neural network models can be trained on
user-supplied peptide activity datasets, enabling the fine-tuning of ligand potency
for dual molecular targets.

### Installation
______________________________
Please note that this code is using Python 3.8.
______________________________

#### macOS/Unix/Windows
To begin, create virtualenv with ```conda```:

```conda create -n PeptideModels python=3.8```

and activate it:

```conda activate PeptideModels```

Install ```pip```:

``` conda install pip ```

and project requirements: 

``` pip install -r requirements.txt```

Then install the ```PeptideModels``` package by running:
```pip install -e .``` at the ```setup.py``` location level.

> [!IMPORTANT] 
> To access ```PeptideModels``` environment from Jupyter Notebook and to use
provided tutorials, please install ```Notebook Conda``` by running:

```conda install -c conda-forge nb_conda```
_____________________________

## Usage
Firstly, an ensemble of multi-task convolutional neural networks is trained using 
peptide sequence data to learn the relationship between the peptide sequences and
their potency levels at two human GPC receptors simultaneously. 
This trained ensemble is subsequently utilised to predict the potencies 
of unknown sequences against human GCG and GLP-1 receptors. Finally, ligand 
optimization is conducted to tailor the activity profile towards one of the three
desired outcomes: selective potency at GCGR, selective potency at 
GLP-1R, or high-potency at both receptors.

![fig](peptide_models/figures/diag.jpg)

### Overview and Tutorials

**Training**:
- To train the ensemble of deep convolutional multi-task neural network models
detailed in our paper for activity prediction across two molecular targets, 
using either the provided human GCG/GLP1 analogous training dataset or your own 
dataset, and please refer to:
```Tutorial-training_ensemble``` in the folder ```notebooks```. 
To run the training, simpy enter ``python train_main.py`` in the command line.
Note that the ensemble training on the provided dataset of 125 peptide sequences takes around 
2 h (processor 2.3 GHz 8-Core Intel Core i9).

**Prediction**:
- To use our pre-trained ensemble for predictions on your dataset of uncharacterised
human GCGR/GLP-1R binding sequences, please refer to the:
```Tutorial-predicting_potencies``` in the ```notebooks``` folder. The running time
for the provided datasets of 288 GCG and GLP-1 orthologs is approximately 10 minutes.

**Optimization**:
- To use our pre-trained ensemble for model-guided ligand optimization, please 
see:```Tutorial-ligand_design``` in the ```notebooks``` directory. The sequence 
optimization process (given the parameters as in the tutorial) is estimated to 
take approximately 25 minutes.

______________________________
> [!IMPORTANT]
> Please note that the results folder and its sub-folders 
> (```results/training```,```results/predictions``` or ```results/ligand_design```) 
> are protected against overwriting. To proceed with a new run of the software, 
> you must firstly manually delete either the entire results folder or the specific
> sub-folder(s) relevant to your task.
______________________________

### Citation
Should you find this repository beneficial to your research, please cite:
> [Machine learning designs new GCGR/GLP-1R dual agonists with enhances biological potency](https://www.nature.com/articles/s41557-024-01532-x)

Puszkarska, A.M., Taddese, B., Revell, J. et al. Machine learning designs new GCGR/GLP-1R
dual agonists with enhanced biological potency. Nat. Chem. (2024).
______________________________

### Paper Supporting Experiments 

The folder ```supporting_experiments``` contains all necessary code and data for replicating the 
supporting findings reported in our paper. This includes the analysis of the training dataset, 
optimization and evaluation of machine learning models, analysis 
of the model-designed compounds, dimensionality reduction, statistical analysis, 
and more. For an in-depth guide on navigating the repository, please refer to
```README_SUPPORT.md```.

-------------------------------
Contact: Anna Puszkarska ( :e-mail: anna.maria.puszkarska21@gmail.com)
