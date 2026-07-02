# Simple Classification Project

This is a simple machine learning project using Python and scikit-learn.

The project trains a classification model to predict whether a customer may face serious financial delinquency within two years using the uploaded dataset.

## Project Structure

```text
simple-github-project/
├── src/
│   └── train.py
├── data/
│   └── breast_cancer.csv
├── notebooks/
│   └── original_notebook.ipynb
├── outputs/
│   ├── metrics.csv
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── models/
│   └── model.pkl
├── README.md
└── requirements.txt
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```

## Output Files

After running the project, the following files will be created:

- `outputs/metrics.csv`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `models/model.pkl`

## Tools Used

- Python
- pandas
- matplotlib
- scikit-learn
- joblib
