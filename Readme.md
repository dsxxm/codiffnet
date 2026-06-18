# DiffNet++

A social recommendation model with gated fusion, contrastive learning, and item-item graph enhancements.

## Highlights

- **Gated Fusion** — adaptively fuses user social embeddings and behavior embeddings
- **Contrastive Learning** — applies contrastive loss on both user and item sides for better representation discrimination
- **Item-Item Graph** — leverages GraphTrust to build item-item structural relationships
- **Ablation switches** — each module can be independently toggled for ablation studies

## Project Structure

```
├── conf/                           # Configuration files
│   ├── flickr_diffnetplus.ini
│   └── yelp_diffnetplus.ini
├── data/                           # Dataset directory
│   ├── flickr/                     # Flickr dataset
│   │   ├── flickr.links            # Social relations
│   │   ├── flickr.rating           # All ratings
│   │   ├── flickr.train.rating     # Training set
│   │   ├── flickr.val.rating       # Validation set
│   │   ├── flickr.test.rating      # Test set
│   │   ├── user_vector.npy         # Pretrained user embeddings
│   │   └── item_vector.npy         # Pretrained item embeddings
│   └── yelp/                       # Yelp dataset (same structure)
├── utils/                          # Utility modules
│   ├── DataModule.py               # Data loading and batching
│   ├── DataUtil.py                 # Dataset initialization and splitting
│   ├── Evaluate.py                 # Evaluation metrics (HR, NDCG)
│   ├── ParserConf.py               # Config file parser
│   └── Logging.py                  # Logging
├── diffnetplus.py                  # Core model implementation
├── train.py                        # Training and evaluation loop
├── entry.py                        # Entry point
├── requirements.txt                # Python dependencies
└── Readme.md
```

## Requirements

- Python 3.7
- TensorFlow 1.15
- See `requirements.txt` for full dependency list

## Datasets

Download from [Google Drive](https://drive.google.com/drive/folders/1YAJvgsCJLKDFPVFMX3OG7v3m1LAYZD5R) and extract into `data/`.

| Dataset | Users | Items | Description |
|---------|-------|-------|-------------|
| Yelp   | 17,237 | 38,342 | Local business ratings with social relations |
| Flickr | 8,358  | 82,120 | Photo ratings with social relations |

## Quick Start

```bash
# Train on Yelp
python entry.py --data_name=yelp --model_name=diffnetplus --gpu=0

# Train on Flickr
python entry.py --data_name=flickr --model_name=diffnetplus --gpu=0
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--data_name` | Dataset name (yelp / flickr) |
| `--model_name` | Model name (currently diffnetplus) |
| `--gpu` | GPU device ID (omit for CPU) |

## Configuration

Config files are in `conf/`, named as `{data_name}_{model_name}.ini`.

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `dimension` | 64 / 32 | Embedding dimension |
| `learning_rate` | 0.0003 | Learning rate |
| `epochs` | 450 | Training epochs |
| `num_negatives` | 16 | Number of negative samples |
| `cl_weight` | 0.01 | Contrastive learning loss weight |
| `use_gated_fusion` | 1 | Enable gated fusion |
| `use_cl` | 1 | Enable user-side contrastive learning |
| `use_item_cl` | 1 | Enable item-side contrastive learning |
| `use_item_item` | 1 | Enable item-item graph |
| `pretrain_flag` | 0 | Load pretrained model |

## Evaluation Metrics

- **HR@K** (Hit Ratio)
- **NDCG@K** (Normalized Discounted Cumulative Gain)

Default evaluation at K = 5, 10, 15.

## References

- Sun et al., "DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation" (IEEE TKDE 2022)
- Original code: https://github.com/PeiJieSun/diffnet
