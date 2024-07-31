<div align="center">
<h2>Seeing the Roads Through the Trees: A Benchmark for Modeling Spatial Dependencies with Aerial Imagery</h2>

[**Caleb Robinson**](https://www.microsoft.com/en-us/research/people/davrob/)<sup>1</sup> · [**Isaac Corley**](https://isaacc.dev/)<sup>2</sup> · [**Anthony Ortiz**](https://www.microsoft.com/en-us/research/people/anort/)<sup>1</sup> · [**Rahul Dodhia**](https://www.microsoft.com/en-us/research/people/radodhia/)<sup>1</sup> · [**Juan M. Lavista Ferres**](https://www.microsoft.com/en-us/research/people/jlavista/)<sup>1</sup> · [**Peyman Najafirad (Paul Rad)**](https://scholar.google.com/citations?user=uoCn8c8AAAAJ&hl=en)<sup>2</sup>

<sup>1</sup>Microsoft AI for Good Research Lab&emsp;&emsp;&emsp;&emsp;<sup>2</sup>University of Texas at San Antonio

<a href="https://arxiv.org/abs/2401.06762"><img src='https://img.shields.io/badge/arXiv-Seeing%20the%20Roads%20Through%20the%20Trees-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/datasets/torchgeo/ChesapeakeRSC/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ChesapeakeRSC%20Dataset-yellow'></a>
</div>

<p align="center">
    <img src="./assets/sample.png" width="800"/><br/>
    <b>Figure 1.</b> Example images and labels from the dataset. Labels are shown over the corresponding NAIP aerial imagery with the "Road" class colored in blue and the "Tree Canopy over Road" class in red.
</p>

We introduce a novel remote sensing dataset for evaluating a geospatial machine learning model's ability to learn long range dependencies and spatial context understanding. We create a task to use as a proxy for this by training models to extract roads which have been broken into disjoint pieces due to tree canopy occluding large portions of the road.

The dataset consists of 30,000 RGBN [NAIP](https://naip-usdaonline.hub.arcgis.com/) images and land cover annotations from the [Chesapeake Conservacy](https://www.chesapeakeconservancy.org/) containing significant amounts of the *"Tree Canopy Over Road"* category.

Models are trained to perform semantic segmentation to extract roads from the background but are additionally evaluated by how they perform on the *"Tree Canopy Over Road"* class. Furthermore, we weight each *"Tree Canopy Over Road"* pixel based on the L1 distance to the nearest *"Road"* pixel resulting in a distance-weighted recall (DWR) metric which we propose as a better proxy for long range modeling performance.

### Reproducing the dataset

We have included the `download_dataset.py` script that demonstrates how we created the aligned NAIP / land cover patches. This script uses the pre-sampled locations in `data/patches.gpkg` and the Maryland land cover dataset from [here](https://www.sciencebase.gov/catalog/item/633302d8d34e900e86c61f81) (it expects the `data/md_lc_2018_2022-Edition/md_lc_2018_2022-Edition.tif` to exist).

### Training

We provide a `train.py` script for reproducing experiments in the paper.

See below for `train.py` usage and arguments:

```bash
usage: train.py [-h] [--batch_size BATCH_SIZE] [--model {deeplabv3+,fcn,custom_fcn,unet,unet++}] [--num_epochs NUM_EPOCHS]
                [--num_filters NUM_FILTERS]
                [--backbone {resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d}] [--lr LR] [--tmax TMAX]
                [--experiment_name EXPERIMENT_NAME] [--gpu_id GPU_ID] [--root_dir ROOT_DIR]

Train a semantic segmentation model.

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Size of each mini-batch.
  --model {deeplabv3+,fcn,custom_fcn,unet,unet++}
                        Model architecture to use.
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for.
  --num_filters NUM_FILTERS
                        Number of filters to use with FCN models.
  --backbone {resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d}
                        Backbone architecture to use.
  --lr LR               Learning rate to use for training.
  --tmax TMAX           Cycle size for cosine lr scheudler.
  --experiment_name EXPERIMENT_NAME
                        Name of the experiment to run.
  --gpu_id GPU_ID       GPU ID to use (defaults to all GPUs if none).
  --root_dir ROOT_DIR   Root directory of the dataset.
```

### Evaluation and figures

We provide an `eval.py` script for evaluating a pretrained checkpoint on the test set. The `notebooks` directory contains jupyter notebooks for reproducing the figures.

See below for `eval.py` usage and arguments:

```bash
usage: eval.py [-h] --model_fn MODEL_FN [--three_class] [--gpu GPU] [--eval_set {test,val}] [--quiet]

options:
  -h, --help            show this help message and exit
  --model_fn MODEL_FN   Model checkpoint to load
  --three_class         Whether to use three classes metrics
  --gpu GPU             GPU to use for inference (default: 0)
  --eval_set {test,val}
                        Which set to run over
  --quiet               Whether to use TQDM progress bar
```

## Citation

If you use this dataset in your work please cite our paper:

```
@article{robinson2024seeing,
  title={Seeing the roads through the trees: A benchmark for modeling spatial dependencies with aerial imagery},
  author={Robinson, Caleb and Corley, Isaac and Ortiz, Anthony and Dodhia, Rahul and Ferres, Juan M Lavista and Najafirad, Peyman},
  journal={arXiv preprint arXiv:2401.06762},
  year={2024}
}
```
