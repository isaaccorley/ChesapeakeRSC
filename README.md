# Chesapeake Roads Spatial Context (RSC)

Code and experiments for the paper, ["Seeing the Roads Through the Trees: A Benchmark for Modeling Spatial Dependencies with Aerial Imagery"]() which introduces the **Chesapeake Roads Spatial Context (RSC) dataset**.

## Download

The dataset is available in HuggingFace Datasets and can be downloaded [here](https://huggingface.co/datasets/torchgeo/ChesapeakeRSC/)

## Dataset

![sample](./assets/sample.png)

We introduce a novel remote sensing dataset for evaluating a geospatial machine learning model's ability to learn long range dependencies and spatial context understanding. We create a task to use as a proxy for this by training models to extract roads which have been broken into disjoint pieces due to tree canopy occluding large portions of the road.

The dataset consists of 30,000 RGBN [NAIP](https://naip-usdaonline.hub.arcgis.com/) images and land cover annotations from the [Chesapeake Conservacy](https://www.chesapeakeconservancy.org/) containing significant amounts of the *"Tree Canopy Over Road"* category.

Models are trained to perform semantic segmentation to extract roads from the background but are additionally evaluated by how they perform on the "Tree Canopy Over Road" class. Furthermore, we weight each "Tree Canopy Over Road" pixel based on the L1 distance to the nearest "Road" pixel resulting in a distance-weighted recall (DWR) metric which we propose as a better proxy for long range modeling performance.
