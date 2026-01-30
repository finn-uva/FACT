# Reproducibility Study of “fairGNN-WOD: Fair Graph Learning Without Demographics”

This repository presents a reproduction of [fairGNN-WOD: Fair Graph Learning Without Complete Demographics](https://www.ijcai.org/proceedings/2025/63). 

## Requirements

Run this command to install the requirements in a conda environment:

```setup
conda env create --name fact_ai --file=fairgnn_wod.yml
```

## File structure

  ```
    .
    ├── fairgnn_model
    ├── vae_model
    ├── README.md
    ├── data.py
    ├── evaluate.py
    ├── fairgnn.py
    ├── fairgnn_wod.yml
    ├── graphvae.py
    ├── main.py
    ├── metrics.py
    ├── train.py
    └── results.ipynb
  ```

## Reproducing the experiments

To compute the evaluation results you can run the commands stated below for the different datasets.

### The following lines reproduce the evaluation of the four classic GNN models (Table 1.)
Note: make sure you are in the code folder

```
python main.py --dataset pokec_z --alpha 0.01 --beta 4 --node 102 --edge 50 --before --device 0 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset pokec_n --alpha 0.01 --beta 4 --node 87 --edge 50 --before --device 1 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'

python main.py --dataset dblp --alpha 0.1 --beta 8 --node 32 --edge 24 --epochs 500 --before --device 2 --models 'GCN' 'GraphSAGE' 'APPNP' 'SGC'
```

## Sources:
The code used to reproduce the results of NIFA comes from the following repository [https://github.com/CGCL-codes/NIFA](https://github.com/CGCL-codes/NIFA). This repository is licensed under CC BY-NC-ND 4.0. However, the authors have explicitly granted us permission to modify and extend the code for our research purposes.

As the NIFA repository does not contain code for reproducing all results, we added code from the repositories below. More information about the code and their licenses can be viewed in the respective repositories.

FA-GNN: [https://github.com/mengcao327/attack-gnn-fairness](https://github.com/mengcao327/attack-gnn-fairness)

FairGNN: [https://github.com/EnyanDai/FairGNN](https://github.com/EnyanDai/FairGNN)

Fairsin: [https://github.com/BUPT-GAMMA/FairSIN](https://github.com/BUPT-GAMMA/FairSIN)

Fairvgnn: [https://github.com/yuwvandy/FairVGNN](https://github.com/yuwvandy/FairVGNN)

TDGIA: [https://github.com/THUDM/tdgia](https://github.com/THUDM/tdgia)
