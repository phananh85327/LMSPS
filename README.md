# Long-range Meta-path Search on Large-scale Heterogeneous Graphs (LMSPS) 🚀
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/pdf/2307.08430)
</div>

<div>
  <p align="center" style="font-size: larger;">
    <strong>NeurIPS 2024 Accepted</strong>
  </p>
</div>

<br>

## News

* **2024-09:** LMSPS is accepted by **NeurIPS 2024**.
* **2024-02:** [Rank 1](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag) on OGBN-MAG Leaderboard.

## Requirements

All experiments are conducted on Tesla V100 GPU (16GB).

Please install the requirements using the following command. (The python version is 3.9.18)

> Windows / RTX 40-series note (thesis fork): see the comments at the top of
> `requirements.txt` -- torch 2.1.2+cu121, torch_sparse 0.6.18 from the PyG index,
> dgl 2.2.1 (the only 2.x release with Windows wheels) + torchdata 0.7.1. Keep
> Python 3.9 to reuse the prebuilt `sparse_tools` extension; link prediction in
> `hgb/` does not need `sparse_tools` at all.

```setup
pip install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
cd ..
```

## Data preparation

For HGB datasets:

```
sh download_hgb_datasets.sh
```

For experiments on the large dataset ogbn-mag, the dataset will be automatically downloaded from OGB challenge.

## Run LMSPS

You can run LMSPS on HGB and ogbn-mag based on the command in [hgb](https://github.com/JHL-HUST/LMSPS/tree/main/hgb) and [ogbn-mag](https://github.com/JHL-HUST/LMSPS/tree/main/ogbn), respectively.

## Cite

If you use LMSPS in a scientific publication, we would appreciate citations to the following paper:

```
@misc{li2024longrange,
      title={Long-range Meta-path Search on Large-scale Heterogeneous Graphs}, 
      author={Chao Li and Zijie Guo and Qiuting He and Hao Xu and Kun He},
      year={2024},
      eprint={2307.08430},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Acknowledgment

This repository benefit from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).
