# Mode Choice Benchmarking Sandbox (MCBS) User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset Management](#dataset-management)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Benchmarking](#benchmarking)
7. [Visualization](#visualization)
8. [Command-Line Interface](#command-line-interface)

## Introduction

MCBS is a toolkit for benchmarking machine learning and econometric models on transportation mode choice datasets. This guide will walk you through the main features and usage of MCBS.

## Installation

[Include detailed installation instructions here]

## Dataset Management

MCBS provides a `DatasetLoader` class for easy management of datasets:

```python
from mcbs.datasets.loader import DatasetLoader

loader = DatasetLoader()

# List available datasets
datasets = loader.list_datasets()
print(datasets)

# Get information about a specific dataset
info = loader.get_dataset_info("example_dataset")
print(info)

# Load a dataset
X, y = loader.load_dataset("example_dataset")
```

## Data Preprocessing



## Model Estimation and Evaluation



## Benchmarking



## Visualization



## Command-Line Interface


For more detailed information on any of these topics, please refer to the inline documentation in the source code or open an issue on our GitHub repository.