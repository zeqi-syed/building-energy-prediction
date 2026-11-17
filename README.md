# Building Energy Efficiency Prediction

Predicting how much energy a building needs for heating and cooling, from its shape and design.

## The question
Using 768 simulated buildings, can we predict heating load and cooling load from features such as height, roof area, wall area and glazing? Which design features matter most?

## What I did
- Checked data quality: no missing values and no duplicates. Dropped Surface Area, which repeats information already in wall and roof area
- One-hot encoded the categorical features (orientation and glazing distribution)
- Explored the data with correlations, histograms, scatter plots and pair plots
- Used PCA (3 components explain 95% of the variance) and K-Means clustering (2 clear groups: low-rise and high-rise buildings)
- Trained Linear Regression, KNN and XGBoost for heating, cooling and both targets together, with grid search and overfitting checks

## Results
XGBoost was the best model for every target:

| Model | Heating R² | Cooling R² | Both R² |
|---|---|---|---|
| Linear Regression | 0.90 | 0.84 | 0.87 |
| KNN | 0.91 | 0.91 | 0.90 |
| XGBoost | **0.998** | **0.97** | **0.98** |

- XGBoost keeps an R² of 0.998 for heating using only two features: relative compactness and glazing area. For linear regression, roof area and overall height matter most.
- Clustering separates low-rise buildings (3.5 m) from high-rise ones (7 m). The high-rise group needs more than twice the heating load (31 vs 13).
- Cooling is harder to predict than heating with every model.

## Files
- `energy_prediction.ipynb`: the full analysis, with outputs
- `data/ENB2012_data.csv`: the dataset

## Run it
```bash
pip install -r requirements.txt
jupyter notebook energy_prediction.ipynb
```

## Data
Energy Efficiency dataset (Tsanas & Xifara, 2012), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/242/energy+efficiency
