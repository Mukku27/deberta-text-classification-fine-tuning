# High-Performance DeBERTa-v3 Fine-Tuning for Text Classification

This repository contains a Jupyter Notebook demonstrating an advanced workflow for fine-tuning the `DeBERTa-v3-large` model for a binary text classification task. The project leverages high-performance training techniques, including multi-GPU support, mixed-precision (FP16), and gradient accumulation, all managed within a custom trainer class.

## Key Features

-   **Model:** Fine-tunes `Microsoft's DeBERTa-v3-large`, a powerful transformer model for natural language understanding.
-   **Custom Trainer:** Includes a robust `AdvancedTrainer` class built with PyTorch for complete control over the training loop.
-   **Multi-GPU Training:** Automatically utilizes all available GPUs using `torch.nn.DataParallel` to accelerate training.
-   **Memory & Speed Optimization:** Implements FP16 (mixed-precision) training via `torch.cuda.amp` and gradient accumulation to handle large models and batch sizes on consumer or prosumer-grade GPUs.
-   **Comprehensive Evaluation:** Calculates and tracks multiple key metrics, including F1-score, Average Precision (AP), ROC AUC, and a custom false-positive score.
-   **Reproducibility:** The notebook is self-contained and sets a random seed for reproducible results.

## Dataset

The notebook is designed to work with a dataset of positive and negative text samples provided as Python pickle files. The training pipeline downloads pre-processed data from the Hugging Face Hub resource `sergak0/sn32`, which contains:
*   `train_pos_list.pickle`: A list of texts for the positive class (label 1).
*   `train_neg_list.pickle`: A list of texts for the negative class (label 0).

The script creates a balanced dataset by randomly sampling an equal number of positive and negative examples before splitting them into training and validation sets.

## Getting Started

Follow these instructions to set up the environment and run the project.

### Prerequisites

-   Python 3.9+
-   NVIDIA GPU with CUDA support (for GPU acceleration)
-   `wget` command-line tool

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mukku27/deberta-text-classification-fine-tuning.git
    cd deberta-text-classification-fine-tuning
    ```

2.  **Install the required Python packages:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    

3.  **Download Model Weights and Data:**
    The notebook includes cells to perform these downloads. The necessary files are:
    -   DeBERTa-v3-large model weights (`deberta-v3-large-hf-weights.zip`)
    -   Fine-tuned state dictionary (`deberta-large-ls03-ctx1024.pth`)
    -   Training data (`data.zip`)

    You can run the download and extraction cells within the notebook or execute the following commands in your terminal:
    ```bash
    # Download files
    wget https://huggingface.co/sergak0/sn32/resolve/main/deberta-v3-large-hf-weights.zip
    wget https://huggingface.co/sergak0/sn32/resolve/main/deberta-large-ls03-ctx1024.pth
    wget https://huggingface.co/sergak0/sn32/resolve/main/data.zip

    # Unzip the archives
    unzip deberta-v3-large-hf-weights.zip -d deberta-v3-large-hf-weights
    unzip data.zip -d data
    ```

## Usage

The entire process is contained within the provided Jupyter Notebook (`.ipynb` file).

1.  Launch Jupyter Lab or Jupyter Notebook.
2.  Open the notebook.
3.  Run the cells sequentially from top to bottom.

The notebook will handle:
-   Installation of dependencies.
-   Downloading and extracting model/data files.
-   Preprocessing the data and creating dataloaders.
-   Initializing the model, optimizer, and scheduler.
-   Running the training for one full epoch.
-   Evaluating the model on the validation set.
-   Saving the best-performing model checkpoint to `best_model.pth`.



## Results

After one epoch of training, the model achieves the following performance on the validation set, demonstrating high accuracy and reliability.

| Metric    | Score  |
| :-------- | :----- |
| `avg_score` | 0.9979 |
| `f1`        | 0.9971 |
| `fp_score`  | 0.9973 |
| `ap`        | 0.9993 |
| `auc`       | 0.9996 |
| `val_loss`  | 0.0145 |

*`avg_score` is a composite metric calculated as `(f1 + fp_score + ap) / 3`.*

## Author

-   **GitHub:** [Mukku27](https://github.com/Mukku27)

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.