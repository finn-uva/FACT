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

### The following lines reproduce the evaluation of our trained models
Note: make sure you are in the code folder

```
# Credit test set results
python evaluate.py --seed=67 --dataset='Credit' --feature_path='./credit/' --edges_path='./credit/' \
     --vae_path='./vae_model/fair_graphvae_model_Credit_67.pt' \
     --gnn_path='./fairgnn_model/fairgnn_wod_model_Credit_67.pt'

# Pokec_n test set results
python evaluate.py  --seed=67 --dataset='Pokec_n' --feature_path='./pokec_n/' --edges_path='./pokec_n/' \
     --vae_path='./vae_model/fair_graphvae_model_Pokec_n_42.pt' \
     --gnn_path='./fairgnn_model/fairgnn_wod_model_Pokec_n_42.pt'

# Pokec_z test set results
python evaluate.py --seed=67 --dataset='Pokec_z' --feature_path='./pokec_z/' --edges_path='./pokec_z/' \
     --vae_path='./vae_model/fair_graphvae_model_Pokec_z_67.pt' \
     --gnn_path='./fairgnn_model/fairgnn_wod_model_Pokec_z_67.pt'
```

### The following lines were used to train the models
```
# Credit
python main.py --seed=67 --dataset='Credit' --feature_path='./credit/' --edges_path='./credit/'

# Pokec_n
python main.py --seed=67 --dataset='Pokec_n' --feature_path='./pokec_n/' --edges_path='./pokec_n/'

# Pokec_z
python main.py --seed=67 --dataset='Pokec_z' --feature_path='./pokec_z/' --edges_path='./pokec_z/'
```


## Sources:
The code used to reproduce the results of NIFA comes from the following repository [https://github.com/CGCL-codes/NIFA](https://github.com/CGCL-codes/NIFA). This repository is licensed under CC BY-NC-ND 4.0. However, the authors have explicitly granted us permission to modify and extend the code for our research purposes.

As the NIFA repository does not contain code for reproducing all results, we added code from the repositories below. More information about the code and their licenses can be viewed in the respective repositories.

FA-GNN: [https://github.com/mengcao327/attack-gnn-fairness](https://github.com/mengcao327/attack-gnn-fairness)

FairGNN: [https://github.com/EnyanDai/FairGNN](https://github.com/EnyanDai/FairGNN)

Fairsin: [https://github.com/BUPT-GAMMA/FairSIN](https://github.com/BUPT-GAMMA/FairSIN)

Fairvgnn: [https://github.com/yuwvandy/FairVGNN](https://github.com/yuwvandy/FairVGNN)

TDGIA: [https://github.com/THUDM/tdgia](https://github.com/THUDM/tdgia)
