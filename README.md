# xjco2121-data-mining-report - Emotion Classification with BERT

## Project Overview
This repository implements a BERT-based multi-label emotion classification system trained on the GoEmotions dataset. The model can classify text into 28 different emotion categories simultaneously.

## Features
- Full fine-tuning of BERT model for multi-label emotion classification
- Support for both original GoEmotions and augmented GoEmotions datasets
- Custom training loop with precision, recall, and F1 score metrics
- Automatic model checkpointing and best model selection
- GPU acceleration support with mixed precision training (FP16)

## Repository Structure
```
├── data/
│   ├── goemotions/          # Original GoEmotions dataset
│   └── goemotions_augmented/ # Augmented version of GoEmotions
├── preprocess.py        # Data loading and preprocessing
├── model.py            # BERT model definition
├── train.py            # Training script
├── plot.py            # plot script
├── outputs/            # Model checkpoints and saved information
├── trainer_state_origin.json # Used for ploting origin result graphs
├── trainer_state_augmented.json # Used for ploting augmented result graphs
└── README.md             # This file
```

## Environment Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/DT-1997/xjco2121-data-mining-report.git
   cd xjco2121-data-mining-report
   ```

2. **Install dependencies**
   ```bash
   pip install "datasets>=3.4.1" huggingface_hub hf_transfer transformers torch numpy scikit-learn
   ```

## Training Process
The training process consists of several key steps:

1. **Data Preprocessing**
   - Loads either the original or augmented GoEmotions dataset
   - Tokenizes text using BERT tokenizer
   - Converts emotion labels to multi-hot vectors
   - Splits data into train/validation/test sets

2. **Model Architecture**
   - Uses BERT-base-uncased as the base model
   - Adds a classification head for 28 emotion labels
   - Configures for multi-label classification

3. **Training Configuration**
   - Batch size: 16 for training, 32 for evaluation
   - Learning rate: 2e-5
   - Number of epochs: 10
   - Mixed precision training (FP16)
   - Automatic model checkpointing
   - Best model selection based on macro F1 score

4. **Evaluation Metrics**
   - Macro Precision
   - Macro Recall
   - Macro F1 Score

## Usage

### Training
To start training, simply run one of the following commands:

```bash
# Train on original GoEmotions dataset
python train.py goemotions

# Train on augmented GoEmotions dataset
python train.py goemotions_augmented
```

The training script will:
1. Load and preprocess the selected dataset
2. Initialize the BERT model
3. Train for 10 epochs
4. Save the best model based on validation F1 score
5. Evaluate on the test set
6. Save the final model and tokenizer

### Output
- Model checkpoints are saved in `outputs/full_ft_{dataset_name}/`
- Training logs are saved in `logs/{dataset_name}/`
- The best model is saved as `outputs/full_ft_{dataset_name}/final_model/`

## Model Performance
The model is evaluated using macro-averaged precision, recall, and F1 score across all emotion labels. The final metrics are printed after training completion.

## Notes
- Training requires a GPU with sufficient VRAM (recommended: 8GB+)
- The training process uses mixed precision (FP16) for faster training
- The model automatically handles multi-label classification
- Checkpoints are saved after each epoch
- The best model is selected based on validation F1 score
