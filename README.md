# xjco2121-data-mining-report - EMO-MTL Pilot Study

## Project Overview
This repository implements a **multi-task fine-tuning** pilot study using **BERT + LoRA** to perform both **multi-label emotion classification** and **emotion intensity regression** on English social media text. We share a single encoder with two heads (classification & regression), train with a weighted loss, and benchmark on **SemEval-2018 Task 1** and **GoEmotions** datasets.

## Repository Structure

```
├── data/
│ ├── goemotions/ # GoEmotions raw & preprocessed JSONL
│ └── semeval2018/ # SemEval-2018 Task 1 raw CSV & JSONL
├── configs/
│ └── unsloth_config.json # Unsloth multi-task finetune config
├── src/
│ ├── preprocess.py # Data loading & preprocessing
│ ├── model.py # Multi-task BERT model with LoRA
│ └── evaluate.py # Custom metrics & evaluation script
├── outputs/ # Checkpoints & best models
├── logs/ # Training logs
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Environment Setup
1. **Clone the repo**  
   ```bash
   git clone https://github.com/DT-1997/xjco2121-data-mining-report.git
   cd xjco2121-data-mining-report
