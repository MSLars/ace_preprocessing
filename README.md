# ace_preprocessing

## Data

Download the original ACE 2005 data from https://catalog.ldc.upenn.edu/LDC2006T06.
Extract (unzip) data, you should receive a folder with a name similar to `ace_2005_td_v7`.

Copy the `ace_2005_td_v7` folder under the folder `ace_raw_data`.

The resulting project structure looks like this:

```
├── ace_raw_data
│   ├── ace_2005_td_v7
│   │   ├── data (* our focus)
│   │   │   ├── Arabic
│   │   │   ├── Chinese
│   │   │   ├── English (* our focus)
│   │   │   │   ├── bc
│   │   │   │   │   ├── adj
│   │   │   │   │   ├── fp1
│   │   │   │   │   ├── fp2
│   │   │   │   │   ├── timex2norm (* our focus)
│   │   │   │   │   │   ├── <file_id>.ag.xml
│   │   │   │   │   │   ├── <file_id>.apf.xml (* our focus, annotations)
│   │   │   │   │   │   ├── <file_id>.sgm.xml (* our focus, raw text)
│   │   │   │   │   │   ├── <file_id>.tab.xml
│   │   │   │   ├── ..
│   │   ├── docs
│   │   ├── dtd
│   ├── __init__.py 
├── data_split
├── ...
```

## Environment

Create a conda environment from the `env.yml` file. This process is
faster with [mamba](https://github.com/mamba-org/mamba)

### Conda Verison
```bash
conda env create -f env.yml

conda activate ace_preprocessing

conda develop .
```
This might take a few minutes.

### Mamba Version

```bash
mamba env create -f env.yml

conda activate ace_preprocessing

conda develop .
```

## Preprocessing for Paper Explaining Relation Classification Models with Semantic Extents

To reproduce the results from the paper, execute following python command. The resulting
files will be in folder `preprocessed_relation_classification_data`.

By default, we use all documents from the ACE 2005 dataset. 
If you want to exclude some "informal" sources, as typically done in other
preprocessing pipelines, use the parameter `reduced`.

For complete dataset:
```bash
python preprocessing/create_relation_classification_samples.py
```

For the reduced dataset:
```bash
python preprocessing/create_relation_classification_samples.py --reduced
```

## Status of Project

As for today (July 2023) this repo helps to reproduce paper results.
We plan to extend the functionality and provide a useful framework to preprocess the ACE 05
dataset in the near future.