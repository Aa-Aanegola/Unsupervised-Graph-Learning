# Centrality Aware Augmentations for Self-Supervised Graph Learning

## Abstract
Self-supervised learning on graphs has gained signif- icant traction due to its ability to learn meaningful node represen- tations without labeled data. A critical component of these meth- ods is data augmentation. Existing approaches predominantly rely on uniform edge dropout, ignoring important structural and contextual variations. We propose centrality-aware edge dropout and adaptive edge addition techniques to enhance graph augmentations for self-supervised learning on graphs. Our meth- ods leverage node centrality metrics to dynamically adjust edge dropout probabilities and add edges, improving representation learning, particularly for underrepresented low-centrality nodes. We validate our techniques on two state-of-the-art frameworks, BGRL and GRACE, across diverse datasets, evaluating per- formance on node classification, node similarity search, and group fairness. Results demonstrate consistent improvements, with eigen-centrality and two-hop sampling emerging as key contributors to the success of our augmentations in addition to accounting for node centrality. This study underscores the importance of structure-aware strategies in advancing graph self- supervised learning and offers a scalable pathway for improving fairness and representation quality with minimal effort.


## Dependencies

- torch
- torch_scatter
- torch_geometric



## Training

All the configuration files can be found in [config](./config). And use the following command to train on the Computer dataset:

```bash
python train.py --flagfile=config/amazon-computers.cfg
```

Flags can be overwritten:

```bash
python train.py --flagfile=config/amazon-computers.cfg --tau=1.0
```



## Acknowledgements
The code is implemented based on [bgrl](https://github.com/nerdslab/bgrl) and [grace](https://github.com/CRIPAC-DIG/GRACE).
