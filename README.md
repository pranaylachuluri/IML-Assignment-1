# Data Cleaning & Preprocessing Pipeline

Exploratory data analysis and preprocessing on two tabular datasets (Palmer Penguins and Diamonds), ending with a reusable preprocessing class. Built for the Introduction to Machine Learning course at the University at Buffalo (Fall 2025).

## What the notebook does

**Penguins dataset**
- Profiles the data: shapes, types, summary statistics and missing values per column
- Fills missing values (mode for categorical columns, mean for numeric ones) and cleans inconsistent text labels
- Finds outliers with z-scores (|z| > 3) and replaces them with the column median
- Visualises species counts, a correlation heatmap, bill length vs body mass by species, and flipper length by species
- Selects features by correlation with two targets (sex and island) and drops weakly correlated ones (|r| < 0.05)
- One-hot encodes categorical columns and min-max scales numeric ones

**Reusable pipeline**
- `OutputPipeline`, a small class with `fit_transform(df, numeric_cols, categorical_cols)` that applies the same imputation, outlier handling, encoding and scaling to any table
- Applied to the Diamonds dataset to show it works on new data

## Tech

Python, pandas, NumPy, Matplotlib, Seaborn

## Run it

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook data_preprocessing.ipynb
```

Put `penguins.csv` and `diamond.csv` in the same folder as the notebook first.
