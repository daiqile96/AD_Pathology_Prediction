# Background

Alzheimer's disease (AD) pathology begins years before clinical symptoms appear, offering a crucial window for early intervention. Predicting brain pathology using clinical features could enable timely interventions to potentially slow or prevent cognitive decline.

## Data

The study uses data from the ROS/MAP cohort, which includes approximately 3,700 participants who were free of dementia at baseline and followed annually. 
- **Features:** Collected clinical features include cognitive test scores, underlying health conditions, and demographic variables.
- **Outcome:** Pathology profiling includes:
  - Amyloid
  - Tangles
  - Gpath
  - NIA-Reagan scores

---

## Method
The analysis consists of the following steps:

1. **Data Exploration:**
   - Performed correlation analysis to identify important features.
   - Identified 40 features with Pearson correlation coefficients > 0.4.
   - Visualized results using a correlation heatmap.

2. **Data Preparation:**
   - Split the data into 80% training and 20% testing sets.
   - Scaled features to ensure compatibility with the regression model.

3. **Baseline Model: Elastic Net Regression:**
A generalized linear model (GLM) with Elastic Net regularization was developed using clinical features from participants' last visits to predict brain pathology. This serves as the baseline for comparison. See full scripts for baseline model [here](https://nbviewer.org/github/daiqile96/AD_Pathology_Prediction/blob/main/elastic_net.ipynb).
   - **Hyperparameter Tuning:** Employed GridSearchCV to optimize hyperparameters.
   - **Model Fitting:** Trained the model with the best parameters identified through tuning.
   - **Evaluation:** Evaluated model performance on the testing dataset.

4. **LSTM**:
An LSTM (Long Short-Term Memory) model was developed to leverage the full longitudinal data. The LSTM's performance was compared to the baseline GLM model to evaluate its effectiveness in capturing temporal patterns.


## Results
### Correlation between features and outcomes
To better understand the relationships among features, a filtered correlation matrix was generated, showing only correlations with an absolute value greater than 0.4:

![](imgs/correlation_between_features.png)

### PCA Analysis and Loadings
Principal Component Analysis (PCA) was applied to reduce the dimensionality of the feature set. Below is the visualization of PCA loadings, showing how features contribute to the first two principal components. Only features with loadings > 0.4 were shown in the figure.

![](imgs/pca_loadings.png)

- Strong Contributors to PC1: 
  - Features like cts_mmse30, motor_handstreng, motor_dexterity, and motor_gait have high positive loadings on PC1. This suggests that PC1 likely represents motor and cognitive function metrics, as these features are related to mobility and cognitive assessment.
- Strong Contributors to PC2: 
  - Features such as dm_cum, diabetes, fx_risks_sum, and hypertension_cum have high positive loadings on PC2. This indicates that PC2 may capture health-related risk factors, such as cumulative diabetes and hypertension effects.


### Elastic Net Regression with PCA (20 Principal Components)
Elastic Net Regression was performed using the 20 principal components derived from PCA. The results are as follows:

- **R-squared on Test Data:** 0.1914
  - The model explains approximately 19.14% of the variance in the Gpath outcome using 20 PCs.

### Elastic Net Regression with Raw Features

- The optimal parameters for the **Gpath** model were selected using GridSearchCV:
  - **Alpha:** 0.1
  - **L1 Ratio:** 0.3
- The following plot visualizes the hyperparameter tuning:
  ![](imgs/parameter_tuning_for_elastic_net.png)

- **Model Performance:**
  - **R-squared on Test Data**:0.2619
  - This indicates that the model explains approximately 26.19% of the variance in the Gpath outcome on the test dataset.

