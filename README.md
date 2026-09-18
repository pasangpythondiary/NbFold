# NbFold
NbFold is a deep learning model for nanobody structure prediction that utilizes a protein language model for feature extraction and incorporates OpenMM and Rosetta for structural refinement.

## Environment Setup

 Create and activate a python virtual environment

Then install necessary packages from this file
```
pip install -r requirements.txt
```

_Note_: PyRosetta can be installed following the instructions [here](http://pyrosetta.org/downloads).

## Download NbFold pretrained weight and unzip and copy the weight in trained_models/ensemble/
Trained weight has been given. Download from [here]https://zenodo.org/records/22760546).
1)	Trained weights for evaluating NbFold benchmark and IgFold benchmark



## Common workflows

_Note_: This project is tested with Python 3.7.9


## To predict Nanobody structure using OpenMM 
Put fasta files in nano folder inside data folder. Note the fasta file must contain >:H  for indicating heavy 
```
python predict_nano_openmm.py --single_chain --use_gpu
```
output will be saved in root nano folder

## To predict Nanobody structure using Rosetta
```
python predict_nano.py --single_chain --use_gpu
```

