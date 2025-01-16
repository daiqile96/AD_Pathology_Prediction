- Background
  
    Alzheimer's disease (AD) pathology begins years before clinical symptoms appear, offering a crucial window for early intervention. Predicting brain pathology using clinical features could enable timely interventions to potentially slow or prevent cognitive decline.

- Data
  
    - Features: The study uses data from the ROS/MAP cohort, which includes approximately 3,700 participants who were free of dementia at baseline and followed annually. Collected clinical features include cognitive test scores, underlying health conditions, demographic.... 
    - Outcome: For ~2,000 deceased participants, pathology profiling includes amyloid, tangles, gpath, and NIA-Reagan scores.

- Objective
  
    The goal is to develop predictive models that use longitudinal clinical features to predict brain pathology (amyloid, tangles, gpath, and NIA-Reagan scores), aiding in early detection and intervention for AD.

- Methods
    - Baseline Model: A generalized linear model (GLM) with Elastic Net regularization was developed using clinical features from participants' last visits to predict brain pathology. This serves as the baseline for comparison.

    - An LSTM (Long Short-Term Memory) model was developed to leverage the full longitudinal data. The LSTM's performance was compared to the baseline GLM model to evaluate its effectiveness in capturing temporal patterns.