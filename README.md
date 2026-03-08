# Hierarchical-Enzyme-Classification-with-Integrated-Sequence-and-Folding-Derived-Representations
This is the implementation code for HiEC
## Reproducing Results

### 1. Testing (Inference)
To reproduce the testing results using the pretrained model:

1.  **Preprocess the sequences:**
    Run the feature extraction ```feature.py``` on the testing dataset ```./Data/testing_seq.fasta```
2.  **Run the evaluation:**
    Execute all cells in `test.ipynb`. The notebook is pre-configured to load the weights from `test_model.pt`.

### 2. Training
To train the model from scratch using the training dataset:

1.  **Preprocess the sequences:**
    Run the feature extraction ```feature.py``` on the testing dataset ```./Data/training_seq.fasta```
3.  **Run the training script:**
    ```python train_egnn_model.py```
## Acknowledgments
This project builds upon the following open-source contributions:

* **Feature Extraction:** Derived from [biomed-AI/GraphEC](https://github.com/biomed-AI/GraphEC).
* **EGNN Architecture:** Based on the implementation by [vgsatorras/egnn](https://github.com/vgsatorras/egnn).

We sincerely thank the authors for their valuable contributions to the research community.
