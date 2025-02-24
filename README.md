# Mini-Project-1


# Employee Satisfaction

## Introduction
We wanted to look into employee productivity and the various factors encompassed. 
This project explores the factors influencing employee productivity and satisfaction, with a focus on:
- Decision Tree model for employee attrition
- Neural Network to predict job satisfaction
- Clustering model for grouping work projects

By combining these approaches, we aim to gain insight into what drives employee engagement and where organizations can make improvements.

# Productivity Research: Satisfaction & Characteristics
- Satisfaction:
   - Productive work environments foster productivity, improve moral, promotes collaboration, and fosters growth  
   - Happy and engaged employees are often more productive
   - Slack's latest annual survey (The State of Work 2023) found that more than 8/10 (82%) said feeling happy and engaged during work was key driver of productivity
   - Microsoft found in the latest work trend index found that lacking clear goals and feeling uninspired were among the top obstacles in the workspace
   - A Univeristy of Oxford's Business School found happy workers to be 13% more productive while the University of Warwick found 12%
   - A Gallup report in 2022 found that 60% of the global population reported feeling emotionally detached at work while 19% claimed to be miserable
- Characteristics:
   - Productive atmosphere
   - Open and Honest communication
   - Compassionate Team Members
   - Positive reinforcement
      - Wage Bonuses
      - Catered Lunches
      - Traveling
      - Raises
      - Reserved Parking
   - Growth Opportunities
   - Positive Thinking
   - Good Work-Life Balance 

### Overview of Neural Network Model ###
This neural network model predicts employee job satisfaction based on key factors including business travel frequency, distance from home, years at company, and age.


Type: Feed-forward neural network with single hidden layer
Input Features:

BusinessTravel (categorical)
DistanceFromHome (numerical)
YearsAtCompany (numerical)
Age (numerical)

Activation Functions:

Hidden Layer: ReLU
Output Layer: Linear

Loss Function: Mean Squared Error (MSE)
Optimizer: Adam

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
  
Side Note:
In the Neural Network code folder, you will find a file named extrater.csv (yes, I know the spelling is incorrect). This file is used by a program I created to extract specific columns of interest, making the data easier to work with. There is also a file called filtered_data.csv, which is the output of the data extraction program. I included this file because it may be helpful for anyone looking to narrow down the data to focus on specific variables.
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
   pip install matplotlib # You might need to get version 3.9 or lower. 3.10 has problems. (visit https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2411 for elaboration)
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
   pip install seaborn
   pip instsallscikit-learn
   pip install matplotlib
   pip install plotly
   pip install kaleido
   pip install numpy
   ```
4. Run the script:
   ```bash
   jupyter notebook project_kmeans_cluster.ipynb

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
   pip install numpy
   pip install seaborn
   ```
4. Run the script:
   ```bash
   jupyter notebook project_tree.ipynb
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

## Neural Network Architecture & Activation Function ##
Activation Function: ReLU (Rectified Linear Unit)
The ReLU activation function is used in the hidden layer of this neural network. It is defined as:

𝑓(𝑥)=max(0,𝑥) <br/>
This means that for any negative input, the output is zero, and for positive inputs, the output is the same as the input. ReLU is commonly used because:

It introduces non-linearity, allowing the network to learn complex patterns.
It helps mitigate the vanishing gradient problem, making training faster and more effective.
Neural Network Layers
This model consists of two layers:
### Hidden Layer ###
8 neurons
ReLU activation function
Takes in an input with the same number of features as the dataset after encoding.
### Output Layer ###
1 neuron
Linear activation function (no activation, just raw output)
Used for regression tasks where the model predicts a continuous value (e.g., job satisfaction score).




# Job Satisfaction Prediction Model Results Analysis

## Overview
This document analyzes predictions from our neural network model that predicts employee job satisfaction based on business travel frequency, distance from home, years at company, and age. The model predicts satisfaction on a scale that appears to range from approximately 2.5 to 7.6.

## Key Findings

### Highest Satisfaction Cases
1. **Test Case 5** (Satisfaction: 7.56)
   - Senior employee (Age: 65)
   - Very long tenure (40 years)
   - Long distance (100 miles)
   - Frequent travel
   
2. **Test Case 7** (Satisfaction: 5.95)
   - Mid-senior employee (Age: 55)
   - Long tenure (25 years)
   - Significant distance (75 miles)
   - Frequent travel

### Lowest Satisfaction Cases
1. **Test Case 3** (Satisfaction: 2.51)
   - Mid-career employee (Age: 35)
   - Moderate tenure (15 years)
   - Moderate distance (20 miles)
   - Frequent travel

2. **Test Case 4** (Satisfaction: 2.65)
   - Young employee (Age: 25)
   - Early career (5 years)
   - Short distance (5 miles)
   - Rare travel

## Pattern Analysis

### Age and Tenure Impact
- **Strong Positive Correlation**: Higher age and longer tenure strongly correlate with higher satisfaction
- Highest predictions occur with employees over 55 years old
- Early-career employees (< 30 years old) consistently show lower satisfaction (2.5-2.8 range)

### Distance from Home Effects
- **Unexpected Pattern**: Longer distances correlate with higher satisfaction when combined with seniority
- Top three satisfaction scores all involve distances > 50 miles
- Short distances (< 10 miles) consistently predict lower satisfaction (2.6-2.8 range)

### Business Travel Impact
- **Mixed Effects**: Travel frequency shows interesting interactions with other variables
- Frequent travelers show highest variability in satisfaction (2.51-7.56)
- Rare travelers show more consistent, moderate satisfaction levels (2.65-4.42)

## Notable Relationships

### Age-Distance Interaction
- Older employees with longer commutes show surprisingly high satisfaction
- This might suggest successful career advancement offsetting commute burden

### Travel-Tenure Relationship
- Frequent travel combined with long tenure predicts higher satisfaction
- May indicate career success and advancement opportunities

## Model Behavior Insights
1. **Range**: Predictions span from 2.51 to 7.56, showing significant variation
2. **Clustering**: Several predictions cluster around 2.6-2.8 for younger employees
3. **Outliers**: Notably high predictions for senior employees with long distances

## Recommendations for Further Investigation
1. Validate the high satisfaction predictions for long-distance senior employees
2. Investigate why younger employees consistently show lower satisfaction
3. Explore additional features that might explain satisfaction variations
4. Consider potential biases in the training data regarding age and tenure

## Technical Notes
- The model shows consistent patterns in prediction behavior
- TensorFlow warnings indicate potential optimization opportunities in the prediction pipeline
- Results suggest strong feature interactions affecting predictions

## Usage Example
```python
test_case = {
    'BusinessTravel': 'Travel_Frequently',
    'DistanceFromHome': 20,
    'YearsAtCompany': 15,
    'Age': 35
}
# Will predict satisfaction in range 2.5-7.6
```

---
# Model Accuracy

Here are the results after running the training script of the neural network. The model decided that 39 epochs was sufficient. At the bottom of the text, the MSE for testing and training is listed. We can see that the testing MSE is ~0.4 more than the training MSE. 
```
Training Monitor Log:
Epoch 1: ✓ New best model!
Epoch 2: ✓ New best model!
Epoch 3: ✓ New best model!
Epoch 4: ✓ New best model!
Epoch 5: ✓ New best model!
Epoch 6: ✓ New best model!
Epoch 7: ✓ New best model!
Epoch 8: ✓ New best model!
Epoch 9: ✓ New best model!
Epoch 10: ✓ New best model!
Epoch 11: ✓ New best model!
Epoch 12: ✓ New best model!
Epoch 13: ✓ New best model!
Epoch 14: ✓ New best model!
Epoch 15: ✓ New best model!
Epoch 16: ✓ New best model!
Epoch 17: ✓ New best model!
Epoch 18: ✓ New best model!
Epoch 19: ✓ New best model!
Epoch 20: ✓ New best model!
Epoch 21: ✓ New best model!
Epoch 22: ✓ New best model!
Epoch 23: ✓ New best model!
Epoch 24: ⚠️ No improvement: patience 1/3
Epoch 25: ✓ New best model!
Epoch 26: ✓ New best model!
Epoch 27: ⚠️ No improvement: patience 1/3
Epoch 28: ✓ New best model!
Epoch 29: ⚠️ No improvement: patience 1/3
Epoch 30: ✓ New best model!
Epoch 31: ⚠️ No improvement: patience 1/3
Epoch 32: ✓ New best model!
Epoch 33: ✓ New best model!
Epoch 34: ✓ New best model!
Epoch 35: ⚠️ No improvement: patience 1/3
Epoch 36: ✓ New best model!
Epoch 37: ⚠️ No improvement: patience 1/3
Epoch 38: ⚠️ No improvement: patience 2/3
Epoch 39: ⛔ Stopping! No improvement for 3 epochs
Final Training MSE: 1.2019
Final Testing MSE:  1.2490
Model saved to neuralnet1.keras
Scaler saved to scaler.pkl
Column list saved to columns.txt
```
### Loss vs. Epoch Graph of Neural Network ### 
![image](https://github.com/user-attachments/assets/32e0a3da-60d4-419f-88f8-adccf73dcf12)


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
- https://www.forbes.com/sites/barnabylashbrooke/2023/07/13/job-satisfaction-is-key-to-workplace-productivity-but-how-do-you-get-it/
- https://www.indeed.com/career-advice/career-development/positive-working-environment

Thank any external contributors, references, or resources that were helpful for the project.



