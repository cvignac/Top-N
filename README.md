# TOP-N: Equivariant set and graph generation without exchangeability

The repository contains code for the paper, accepted at ICLR 2022. 

To run the SetMNISt or CLEVR experiments, download the corresponding repositories (https://github.com/Cyanogenoid/dspn
and https://github.com/LukeBolly/tf-tspn/blob/master/models/set_prior.py) and replace the corresponding file (`dspn.py`
or `tspn.py`) by the one present in the `dspn_tspn` folder.

To run set generation, run `main.py`.

To run graph generation, run `molecule_generation/graph_main.py`.


To cite the paper, you can use the following reference:

```
@inproceedings{
vignac2022topn,
title={Top-N: Equivariant Set and Graph Generation without Exchangeability},
author={Clement Vignac and Pascal Frossard},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=-Gk_IPJWvk}
}
```