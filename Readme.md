# **Project Module 3 - Diamond Price Prediction Challenge 🔮💶💎**

The main reason for this project is to apply **Machine Learning** techniques and data analysis in a real-world context, where diamond valuation is influenced by some factors; by developing an accurate and robust predictive model capable of estimating the price of diamonds based on a set of relevant features.


## 1. Project description 👇
> Project Module 3 - Part time Sept 2023 - [Ironhack Madrid - Data Analytics Bootcamp](https://www.ironhack.com/es-en/data-analytics)

This competition has been created specifically in in [**Kaggle**](https://www.kaggle.com/competitions/ihdatamadpt0923projectm3/overview)for the students of the Ironhack Data Analytics Bootcamp, providing an opportunity to apply Machine Learning concepts and master skills related to.

 **The main challenge** was getting the lowest **RMSE** (Root Mean Squared Error) as the chosen evaluation metric.

 ### Functional architecture design:

![Pipeline Architecture Design](Pipeline%20Architecture.jpg)


- Baseline & Preprocessing: 
    - **LabelEncoding** for both datasets
    - **Feature Engineering with correlation:** The most influencing feature was **Carat** towards the price, and columns=['x','y','z'] were dropped.
    - **Robust Scaled** because is special for outliers, before fitting in the model.



- Training:
    - **XGBoost with Cross Validation and Grid Search:** the hyperparameters were **XGBRegressor**(gamma=0.01, learning_rate=0.1, max_depth=6, n_estimators=200)

- Submission: The lowest **RMSE** I reached was 546 **compared to** Kaggle's RMSE of 533.



 ## **2. Technology stack 💻**

#### Programming language:
- [Python](https://docs.python.org/3/)

#### Datasets:
- `diamonds_train.db`: The training set (imported as .csv from DBeaver)
- `diamonds_test.csv`: The test set.
- `sample_submission.csv`: A sample submission file in the correct format.

#### Python Libraries:
- [matplotlib.pyplot](https://matplotlib.org/stable/contents.html): For data visualization.
- [random](https://docs.python.org/3/library/random.html): For generating random numbers.
- [numpy](https://numpy.org/doc/stable/): For mathematical operations and array manipulation.
- [pandas](https://pandas.pydata.org/docs/reference/frame.html): For data manipulation and analysis.
- [sklearn](https://scikit-learn.org/stable/): For machine learning modeling and model evaluation.

#### Models and Evaluation Metrics:
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): Used for predicting diamond prices.
- Evaluation metrics include: `mean_squared_error`, `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`, `r2_score`.
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): Used for classification.
- [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html): Used for classification problems with neural networks.
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html): Used for classification problems with k-nearest neighbors.

#### Data Preprocessing:
- [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html): For feature scaling.

#### Modeling Tools:
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html): For splitting data into training and testing sets.
- [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html): For model evaluation using cross-validation.
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html): For hyperparameter tuning using cross-validated grid search.


## **4. Main conclusions 📁**

The competitions had its bright sight regarding a non-stop learning, because of the RMSE metric that drove us to master ML skills.



###  **Contact info📧**
For further information, reach me at andrew.bavuels@gmail.com

---