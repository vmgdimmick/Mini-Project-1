# Mini-Project-1


# Employee Satisfaction

## Introduction
We wanted to look into employee productivity and the various factors encompassed. 
This project explores the factors influencing employee productivity and satisfaction, with a focus on:
- Decision Tree model for employee attrition
- Neural Network to predict job satisfaction
- Clustering model for grouping work projects

By combining these approaches, we aim to gain insight into what drives employee engagement and where organizations can make improvements.

## Data Source
Our data is derived from two separate Kaggle datasets, cleaned and merged into a single comprehensive dataset for the clustering model. The rows for both datasets are employee number. The columns of both data sets are listed below.

### Attrition Rate of a Company:
- EmployeeID
- Age
- Attrition
- BusinessTravel
- Department
- DistanceFromHome
- Education
- EducationField
- EmployeeCount
- Gender
- JobLevel
- JobRole
- MartialStatus
- MonthlyIncome
- NumCompaniesWorked
- Over18
- PercentSalaryHike
- StandardHours
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- YearsAtCompany
- YearsSinceLastPromotion
- YearsWithCurrManager
- EnvironmentSatisfaction
- JobSatisfaction
- WorkLifeBalance
- JobInvolvement
- PerformanceRating
   
### Employee Productivity and Satisfaction
- Name
- Age
- Gender
- Projects Completed
- Productivity (%)
- Satisfaction Rate (%)
- Feedback Score
- Department
- Position
- Joining Date
- Salary
  

## Installation and Setup
Instructions for setting up the project environment and running the neural network:
1. Clone this repository:
   ```bash
   git clone https://github.com/CaileyCrandall/Mini-Project-1
   ```
2. Navigate to the project directory:
   ```bash
   cd Mini-Project-1
   ```
3. Install dependencies:
   ```bash
   pip install pandas
   pip install scikit-learn
   pip install tensorflow
   ```
4. Run the script:
   ```bash
   python train.py # (or whatever script you are running)
   ```
 Instructions for setting up the project environment and running the clustering model:
1. Clone this repository:
   ```bash
   git clone https://github.com/CaileyCrandall/Mini-Project-1
   ```
2. Navigate to the project directory:
   ```bash
   cd Mini-Project-1
   ```
3. Install dependencies:
   ```bash
   pip install pandas
   pip install scikit-learn
   pip install tensorflow
   pip install matplotlib # You might need to get version 3.9 or lower. 3.10 has problems. (visit https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2411 for elaboration)
   ```
4. Run the script:
   ```bash
   python train.py # (or whatever script you are running)

Instructions for setting up the project environment and running the decision tree:
1. Clone this repository:
   ```bash
   git clone https://github.com/CaileyCrandall/Mini-Project-1
   ```
2. Navigate to the project directory:
   ```bash
   cd Mini-Project-1
   ```
3. Install dependencies:
   ```bash
   pip install pandas
   pip install scikit-learn
   pip install tensorflow
   ```
4. Run the script:
   ```bash
   python train.py # (or whatever script you are running)

## Data Processing
For the neural network, before analysis, data cleaning steps included:

1. Handling missing values:
We dropped rows containing missing (NaN) values to ensure a clean dataset (e.g., df.dropna(inplace=True)). This was a simple, pragmatic choice since only a small          fraction of rows were affected.

3. Encoding categorical variables where needed.
We used one-hot encoding via pd.get_dummies for columns like "BusinessTravel". This converts each category into a binary feature (0 or 1) so that the model can handle them as numeric inputs.

5. Normalizing or standardizing numerical features for modeling consistency.
We used scikit-learn’s StandardScaler for our numeric columns (e.g., DistanceFromHome, Age, YearsAtCompany). Standardization rescales each feature so it has a mean of 0 and standard deviation of 1, helping the neural network train more efficiently.

**Summary of the Neural Network Creation Process**

1. **Data Collection & Filtering**  
   - We began with a CSV containing employee-related variables (e.g., *BusinessTravel, DistanceFromHome, YearsAtCompany, Age, JobSatisfaction*).  
   - We filtered and cleaned the data so only the required columns remained (removing unused columns and rows with missing values if needed).

2. **Handling Missing Data & Outliers**  
   - To ensure a clean dataset, we dropped rows that contained missing values. Because these rows represented a small fraction of the total, removing them had minimal impact.  
   - We did not specifically remove or transform outliers, as the data did not show substantial extreme values.

3. **Encoding Categorical Variables**  
   - We applied one-hot encoding (via `pd.get_dummies`) to convert any categorical features (e.g., *BusinessTravel*) into numeric indicator variables. This allows the neural network to process them effectively.

4. **Normalization / Standardization of Numerical Features**  
   - We used the `StandardScaler` from scikit-learn on numerical columns (e.g., *DistanceFromHome, Age, YearsAtCompany*). This transformation centers each feature around a mean of 0 and a standard deviation of 1, helping the neural network learn more efficiently.

5. **Choice of Modeling Approach**  
   - Problem Type: We predicted a continuous variable, *JobSatisfaction*, indicating this is a regression task (as opposed to classification).  
   - Neural Network Structure:  
     - We built a simple feed-forward (dense) neural network with a single hidden layer of a moderate number of neurons.  
     - Activation functions in the hidden layer used ReLU, while the final layer had a linear output activation.  
   - Justification: For a continuous target variable with both numeric and categorical features, a small neural network can learn non-linear relationships while remaining relatively easy to set up. Although simpler linear models or tree-based methods could also work, the neural network offers flexibility and can capture more complex interactions if needed.

6. **Training & Evaluation**  
   - Loss Function: We used Mean Squared Error (MSE)—a standard choice for regression tasks.  
   - Optimizer: We employed Adam, a commonly used adaptive optimizer that converges quickly and handles different feature scales well.  
   - Train/Test Split: We split the dataset (80% training, 20% testing) to ensure we could measure generalization.
   - Early stopping: Implemented an early stopping feature to allow for an increase in epochs along with overfitting avoidance.
   - Results: After sufficient number of epochs, we observed the final training and testing MSE to gauge how well the model fit the data without overfitting.
   - Visualization: Also implemented a loss vs. epoch graph to visualize the regression.

7. **Making Predictions**  
   - We saved the trained model and the fitted scaler (via pickle).  
   - For new data, we apply the same one-hot encoding and scaling steps before calling `model.predict(...)`. This ensures consistent preprocessing and valid predictions.

8. **Statistical Considerations**  
   - Regression Analysis: By minimizing MSE, we effectively employed least-squares principles in the neural network framework.  
   - Encoding: One-hot encoding is a standard categorical-encoding method to avoid treating categorical data as continuous.  
   - Data Standardization: Aligning feature distributions helps maintain stable gradients and faster convergence when training neural networks.

Overall, this pipeline—data cleaning, encoding, scaling, model definition, training, and evaluation—is a well-rounded approach to building a predictive neural network for continuous outcomes like *JobSatisfaction*. It balances simplicity and low computational cost (because of the single hidden layer) with enough flexibility to model non-linearities in the data.
   
## Analysis and Discussion
Provide an overview of the methods used for analysis. Include key findings, interpretations, and any challenges encountered.

## Results
Summarize the results with supporting visuals, tables, or explanations.

## Contributions
List the team members and their contributions:
- **Cailey Crandall** - [Role and contributions]
- **Vance Dimmick** - [Role and contributions]
- **Sheridan Traish** - [Role and contributions]
- **Renee Vannice** - [Role and contributions]

## Future Work
Feature Engineering: Investigate additional features (e.g., demographic data, employee engagement survey results) to enhance model accuracy.
Model Optimization: Hyperparameter tuning for improved performance across all models.
Deployment: Packaging models for real-time use within an organization’s HR analytics pipeline.

## Acknowledgments
Thank any external contributors, references, or resources that were helpful for the project.



