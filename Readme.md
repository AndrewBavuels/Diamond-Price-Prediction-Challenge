# **Project Module 3 - Diamond Price Prediction Challenge💻🌐💶**

The main reason for this project is to apply **Machine Learning** techniques and data analysis in a real-world context, where diamond valuation is influenced by some factors; by developing an accurate and robust predictive model capable of estimating the price of diamonds based on a set of relevant features.


## 1. Project description 👇
> Project Module 3 - Part time Sept 2023 - [Ironhack Madrid - Data Analytics Bootcamp](https://www.ironhack.com/es-en/data-analytics)

This competition has been created specifically in in [**Kaggle**](https://www.kaggle.com/competitions/ihdatamadpt0923projectm3/overview)for the students of the Ironhack Data Analytics Bootcamp, providing an opportunity to apply Machine Learning concepts and master skills related to.

 **The main challenge** was getting the lowest **RMSE** (Root Mean Squared Error) as the chosen evaluation metric.

 #### Functional architecture design:

 
 
- Baseline:
- Preprocessing:
- Training:
- Submission:



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




## **3. Minimal Functional App ⚙️**

To know the purpose of our visualization, **first** we must know the audience we will present it to, in order to prepare our visualization and give the right message with the right words.

For this case, I developed a fictional character for the **Profile Persona** named Lucia, which the visualization will allow her to **take decisions about** what skills should be trained for **and** what countries offers the best for her professional development.

![Lucia](https://github.com/AndrewBavuels/ih_datamadpt0923_project_m2/raw/main/images/Lucia_Profile_Persona.png)

With this **Profile Persona**, we can formulate possible questions the people (represented by Lucia) would ask. With the Dashboard (specially if is an static report), we could give answers to them.


> **Note:** "Static" means if the dashboard is like an image.


#### How does it work?
From the **Profile Persona** I formulated the following question they need answer for:

- What positions can I start with in the job market if my background is related to Sales?
- What are the salaries of tech positions related to my career?
- What options do I have to work, and at the same time maintain frequent contact with family and friends?
- How can I immediately ensure my entry into the job market?
- Should I look for work in Spain? In a nearby country? or Remote?

**The following step was** to load the dataset in Tableau and perform my report/dashboard to communicate my insights, as you can see as follows:

![Dashboard](https://github.com/AndrewBavuels/ih_datamadpt0923_project_m2/raw/main/images/BI%20Report-Dashboard.png)

> The previous output is exported in Tableau Public.

At a glance, the **insights** I could get from the Dashboard, based on Lucia's questions, are the following ones:

- It starts with the area Data Analysis, according to her "quote" in the Profile Persona.

- Data Analyst instead of Business Data Analyst, because the first one has 37 job records and more probability of getting a job. **Instead of the Business one role** with just 2 records and a lower salary average than the first one.

- There’s the option to start immediately in Part-time and to take a course in her free time to enhance her skills.

- Most of the Data Analyst roles are in Spain, which **she can afford to start with an In-person job** with an average salary around € 42,000 per year. 

## **4. Main conclusions 📁**

With this report, Lucia made her decision of getting the tools and trainings to enhance her business and sales background to get an entry in the Tech Market as a Data Analyst.

What motivates her the most is staying in Spain because of the high probabilities to achive her goals, along with keeping on touch with family and friends.

This is just the beginning of Lucia professional growth. For more details, [**click here**](https://public.tableau.com/views/Readme_mdProjectM2DataScience/1_Overview?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link) to interact with the dynamic dashboard.

###  **Contact info📧**
For further information, reach me at andrew.bavuels@gmail.com

---