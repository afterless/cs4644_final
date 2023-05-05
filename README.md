# CS 4644 Final Project: Git Re-Basin on 2L MLP trained on Modular Addition

## Purpose/Goal

The purpose of this final project is to replicate the Git Re-Basin [paper](https://arxiv.org/abs/2209.04836) on a 2L MLP trained on modular addition. The goal of this project
would be to build upon Nanda et al.'s [work](https://arxiv.org/pdf/2301.05217.pdf) into interpreting networks that have learned moduler addition via "grokking" as shown in Power et al.'s [work](https://arxiv.org/pdf/2201.02177.pdf). With Git Re-Basin, we hope to show the existence of single basin phenomenon with respect to certain architectures, in this case modular addition and whether there is a connection to properties like grokking and those explored in Git Re-Basin as well as trying to replicate the paper successfully due to previous failed attempts by others. We will be using Stanislav Fort's [replication](https://github.com/stanislavfort/dissect-git-re-basin), the Git Re-Basin [codebase](https://github.com/samuela/git-re-basin), and Neel Nanda's [codebase](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=BhhJmRH8IIvy) as a starting point for this project.


## Setup
If you wish to run this project out of interest or to contribute, you can setup your machine using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or something similar to make a virtual environment. If you are on macOS or Linux you can use the following:

```bash
ENV_PATH=~/cs4644_final/.env/
cd $ENV_PATH
conda create -p $ENV_PATH python=3.10 -y
conda install -p $ENV_PATH pytorch=2.0.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

If you are on Windows, you can run this:

```
$env:ENV_PATH='c:\users\<user_name>\cs4644_final\.env'
cd cs4644_final
conda create -p $env:ENV_PATH python=3.10 -y
conda install -p $env:ENV_PATH pytorch=1.12.0 torchtext torchdata torchvision -c pytorch -y
conda run -p $ENV_PATH pip install -r requirements.txt
```

## Results

### Plots
### Takeaways