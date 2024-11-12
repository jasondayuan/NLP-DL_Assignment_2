# Introduction
Second assignment of NLP-DL (2024~2025 Semester 1).
- Task 1 - dataHelper.py
- Task 2 - train.py
- Task 3 - train_adapter.py

The report of this assignment is in report.pdf.
# How to run
## Task 2
If you want to finetune model {model_name} (e.g. ```roberta-base```) on dataset {dataset_name} (e.g. ```restaurant_sup```), simply run the following command.
```bash
./train_multi_run_{dataset_name}.bash {model_name}
```
The script runs the finetuning script 5 times, with a different seed each time. 
## Task 3
If you want to finetune model {model_name} (e.g. ```roberta-base```) on dataset {dataset_name} (e.g. ```restaurant_sup```) with PEFT, simply run the following command.
```bash
./train_adapter_multi_run_{dataset_name}.bash {model_name}
```
The script runs the PEFT script 5 times, with a different seed each time. 