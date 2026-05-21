
## SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction [ACL 2025]

This repository contains research code for the paper [*SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction*](https://arxiv.org/abs/2411.16765).

<p align="middle">
  <img src="imgs/shubert_pipeline.png"  alt="SHuBERT Overview">
</p>


We introduce SHuBERT (Sign Hidden-Unit BERT), a self-supervised contextual representation model learned from approximately 1,000 hours of  American Sign Language video. SHuBERT adapts masked token prediction objectives to multi-stream visual sign language input, learning to predict multiple targets corresponding to clustered hand, face, and body pose streams. SHuBERT achieves state-of-the-art performance across multiple tasks including sign language translation, isolated sign language recognition, and fingerspelling detection.  

----

## ISL Adaptation Pipeline (Unlabeled Videos)

This repository includes a complete pipeline to continue self-supervised pretraining of a pretrained SHuBERT model on Indian Sign Language (ISL) videos. It provides MediaPipe-based preprocessing, DINOv2 features, k-means clustering, SHuBERT SSL training, embedding extraction, and visualization utilities.

### ISL Pipeline Installation

```bash
pip install -r requirements.txt
```

### ISL Pipeline Configuration

Edit `configs/config.yaml` to set your dataset root, cache directories, and pretrained weights path.

### ISL Pipeline Training

```bash
python train.py
```

### ISL Pipeline Embedding Extraction

```bash
python extract_embeddings.py --video /path/to/video.mp4 --output_dir outputs
```

Outputs:
- `outputs/embeddings.npy` (shape [T, 768])
- `outputs/cluster_map.json`

### ISL Pipeline Visualization

```bash
python visualize_embeddings.py --embeddings outputs/embeddings.npy --cluster_map outputs/cluster_map.json
```

### Installation

We provide installation and inference instructions in [QUICKSTART.md](QUICKSTART.md).

### Usage
#### 1. Preparing the data

We describe how to prepare the datasets in [DATASETS.md](DATASETS.md).


#### 2. Model Weights

Please download the weight of SHuBERT (as well as the DINO Face and Hand) weights [link](https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing).

#### 3. Feature Extraction

We describe how to extract features from the pretrained model in [FEATURES.md](FEATURES.md).


#### 4. Pretraining

- sbatch train_shubert.sh


#### 5. Fine-tuning on Downstream Tasks

TODO

---- 
### Citing our work
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{gueuwou-etal-2025-shubert,
    title = "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction",
    author = "Gueuwou, Shester and Du, Xiaodan and Shakhnarovich, Greg and Livescu, Karen and Liu, Alexander H.",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
}
```


### References
This codebase is heavily influenced by the [DinoSR](https://github.com/Alexander-H-Liu/dinosr) and [Fairseq](https://github.com/facebookresearch/fairseq) repositories.

### License
This project is primarily under the MIT license.
