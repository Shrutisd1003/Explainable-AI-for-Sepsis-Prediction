# Explainable AI for Sepsis Prediction

Sepsis is a severe condition needing quick action. This project introduces a Python machine learning tool. It predicts if someone might get sepsis and explains why. We use advanced methods like XGBoost and Lime to make the predictions clear. This helps healthcare pros make better choices.

## Steps:
1. Exploratory Data Analysis (EDA):
    - Analyze the dataset sourced from Kaggle.
    - Utilize visualizations and statistical summaries to understand data characteristics.
    - Identify key features and trends that inform subsequent modeling decisions.

2. Preprocessing:
    - Handle missing values and remove irrelevant attributes.
    - Address class imbalance through techniques like SMOTE.
    - Evaluate alternative strategies to optimize model performance.

3. Model Training:
    - Train a predictive model using the XGBoost algorithm.
    - Perform hyperparameter tuning to maximize predictive capabilities.
    - Compare different model iterations and select the most effective configuration.

4. Model Pickling:
    - Serialize the trained model using pickling.
    - Ensure compatibility across various environments for seamless deployment and reuse.

5. Deployment on Streamlit:
    - Deploy the model using Streamlit, a user-friendly web application framework.
    - Allow users to input data and receive real-time predictions on sepsis risk.
    - Provide interpretability through visualizations highlighting feature contributions to predictions.

link to dataset: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
