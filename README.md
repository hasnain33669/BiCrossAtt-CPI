# BiCrossAtt-CPI

**BiCrossAtt-CPI** is a deep learning framework for **Compound-Protein Interaction (CPI) prediction**.

## Key Features

- **Graph Neural Networks (GNNs):** Learn structural representations of compounds from molecular graphs.
- **Pretrained ESM-2 Embeddings:** Capture contextual information from protein amino acid sequences.
- **Bidirectional Cross-Attention:** Model interactions between compound atoms and protein residues in both directions.
- **Feature Fusion:** Combine global and cross-attended representations.
- **MLP Classifier:** Predict compound-protein interaction probabilities.

## Model Architecture

The BiCrossAtt-CPI framework consists of the following components:

1. **Compound Encoding:** SMILES strings are converted into molecular graphs and processed using a DenseNet-style architecture with GraphConv layers to obtain atom-level representations.
2. **Protein Encoding:** Protein amino acid sequences are encoded using the pretrained ESM-2 protein language model to generate residue-level embeddings.
3. **Bidirectional Cross-Attention:** Compound atom representations attend to protein residue representations, while protein residues attend to compound atom representations.
4. **Feature Fusion:** Global compound and protein features are combined with cross-attended representations.
5. **MLP Classification:** The fused representation is passed through a multilayer perceptron (MLP) to predict the probability of a compound-protein interaction.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- RDKit
- PyTorch Geometric (`torch_geometric`)
- fair-esm
- Transformers
- scikit-learn
- pandas
- NumPy
- tqdm
- matplotlib
- seaborn


### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

The datasets used in this study were obtained from the benchmark dataset introduced by Liu et al. (2015).

**Reference:**

Liu, H., Sun, J., Guan, J., Zheng, J., & Zhou, S. (2015). Improving compound-protein interaction prediction by building up highly credible negative samples. *Bioinformatics*, 31(12), i221–i229.


## Project Structure

```text
BiCrossAtt-CPI/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── src/
│   ├── __init__.py
│   ├── data_utils.py       
│   ├── models.py           
│   ├── train.py            
│   └── evaluate.py         
└── configs/
    └── config.yaml
```

### Train the Model

```bash
python src/train.py
```
