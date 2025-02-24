# Mini-Project-1 Employee Satisfaction 




## Introduction
We wanted to look into employee productivity and the various factors encompassed. 
This project explores the factors influencing employee productivity and satisfaction, with a focus on:
- Decision Tree model for predicting whether employee will perform well
- Neural Network to predict job satisfaction
- Clustering model for exploring factors that influence number of projects completed

By combining these approaches, we aim to answer our question: How do certain factors influence employee satisfaction and productivity?

# Assumptions and Questions  

## **Assumptions**  
(Based on attrition and employee satisfaction data)  

- People who take fewer vacations tend to perform better at work.  
- Alternatively, fewer vacations may lead to increased stress.  
- A healthy work environment is beneficial to employee productivity.  
- Stress levels can affect work performance.  
- Long commutes and heavy traffic negatively impact employee well-being and productivity.  
- Fewer employee benefits lead to decreased productivity.  
- All projects are assumed to be of the same difficulty level.  
- Everyone is assumed to have the same baseline happiness.  

## **Questions**  

### **General Employee Productivity Factors**  
- How does happiness contribute to employee productivity?  
- How does education level impact work performance?  
- How does time spent at the company affect productivity?  
- How does overall satisfaction influence work efficiency?  

### **Neural Network Analysis**  
- How many projects does an employee complete in a year?  
- How do vacation days affect productivity?  
- How does age correlate with performance?  
- How does tenure at the company impact output?  

### **Clustering Model Analysis**  
- How does work productivity vary based on salary?  
- How does work productivity vary based on satisfaction?
- How does work productivity vary based on years spent at a company?  

### **Decision Tree Analysis**  
- Can we predict if an employee is going to be a top performer?

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


# Data Source
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
# Installation and Setup
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

# Neural Network Model
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

# Decision Tree  

This decision tree model predicts whether employees will perform above or below the 75th percentile for projects completed based on the following features:

- Productivity (%)  (Integer)  
- Satisfaction Rate (%)  (Integer)  
- Department_x  (Categorical)  
- Position  (Categorical)  
- Salary  (Integer)  
- Attrition  (Categorical)  
- BusinessTravel  (Categorical)  
- Department_y  (Categorical)  
- DistanceFromHome  (Integer)  
- Education  (Integer)  
- EducationField  (Categorical)  
- Gender  (Categorical)  
- JobLevel  (Integer)  
- JobRole  (Categorical)  
- MaritalStatus  (Categorical)  
- NumCompaniesWorked  (Float)  
- PercentSalaryHike  (Integer)  
- StandardHours  (Integer)  
- TotalWorkingYears  (Float)  
- YearsAtCompany  (Integer)  
- YearsSinceLastPromotion  (Integer)  
- EnvironmentSatisfaction  (Float)  
- JobSatisfaction  (Float)  
- WorkLifeBalance  (Float)  
- JobInvolvement  (Integer)  

## Performance Metrics  

- Root Mean Squared Error (RMSE)   
- Confusion Matrix Metrics:   
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score

# K-Means Clustering  

This clustering model analyzes relationships between:  

- Projects completed and salary   
- Projects completed and employee satisfaction   
- Projects completed and years an employee has worked at the company   

## Model Details  

-  Type:  K-Means Clustering  
-  Silhouette Score:  0 to 1  


# Data Processing

## Neural Network
For the neural network, before analysis, data cleaning steps included:

1. Handling missing values:
We dropped rows containing missing (NaN) values to ensure a clean dataset (e.g., df.dropna(inplace=True)). This was a simple, pragmatic choice since only a small          fraction of rows were affected.

3. Encoding categorical variables where needed.
We used one-hot encoding via pd.get_dummies for columns like "BusinessTravel". This converts each category into a binary feature (0 or 1) so that the model can handle them as numeric inputs.

5. Normalizing or standardizing numerical features for modeling consistency.
We used scikit-learn’s StandardScaler for our numeric columns (e.g., DistanceFromHome, Age, YearsAtCompany). Standardization rescales each feature so it has a mean of 0 and standard deviation of 1, helping the neural network train more efficiently.

## Decision Tree  

Before analysis, data cleaning steps included:  

- Random Sampling – A dataset was shortened via random sampling to achieve row congruence between datasets.  
- Dataset Merging – The 200-row sampled dataset was merged with another 200-row dataset on an artificially created mutual column, with duplicate columns removed.  
- Binary Target Creation – A new binary column was added to indicate whether employees completed more or less than the 75th percentile (17) projects.  

# Creation Process
## Summary of the Neural Network Creation Process ##

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
## Summary of Decision Tree Creation Process  

## Creation Process  
1.**Data Collection & Filtering**  
We began with a CSV containing employee-related variables, such as:  
 BusinessTravel  
 DistanceFromHome  
 YearsAtCompany  
 Age  
 JobSatisfaction  

2. **Encoding Categorical Variables**  
To handle categorical variables, **LabelEncoder** from the `sklearn.preprocessing` package was used. This function assigned numerical labels to categorical columns, allowing the decision tree to process them as variables.  

3. **Choice of Modeling Approach***  

3.1 Problem Type: 
We predicted a binary variable, Output, indicating whether an employee will perform in the 75th percentile of project completion.  
Since this is a classification task, a decision tree model was selected over clustering or regression.  

3.2 Decision Tree Structure  
A simple decision tree diagram was built, utilizing an unsupervised learning approach.  
The model made predictions on whether an employee would perform in the 75th percentile based on selected independent variables.  

3.3 Justification  
A decision tree is well-suited for predicting a binary target variable when dealing with a mix of categorical and numerical features.  
The model learns from previous observations and applies that knowledge to new instances, making it a useful tool for understanding patterns in employee performance.  

4. **Training and Evaluation**  

4.1 Evaluation Metrics  
Loss Function – Root Mean Squared Error (RMSE) was included for reference but was not the primary metric.  
Confusion Matrix Metrics:  
   - **Accuracy** =  (0.8)– proportion of all correct predictions.  
   - **Precision** = (0.75) – model’s ability to avoid false positives.  
   - **Recall** = (0.5) – model’s ability to correctly identify positive instances.  
   - **F1 Score** = (0.6) – balances the tradeoff between precision and recall.  

4.2 Train/Test Split  
The dataset was split into training and test sets.  
**90% training data (180 observations)** was used to train the model.  
**10% test data (20 observations)** was used to evaluate performance.  

4.3 Key Takeaways  
The most critical metric for this task is Recall, as it measures the model's ability to correctly identify employees who will perform in the 75th percentile.  
However, our model's Recall score is relatively low, suggesting that it still struggles to correctly classify high-performing employees.  
This indicates that further refinements are needed to improve predictive accuracy.  


# Explaining the math
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

## Decision Tree Explanation

A Decision Tree works by recursively splitting the dataset based on feature values that maximize information gain or minimize impurity. At each step, the algorithm selects the feature that best separates the data into distinct groups. The final tree consists of decision nodes and leaf nodes, where each path from root to leaf represents a decision rule.

For evaluation, the Root Mean Squared Error (RMSE) measures the difference between predicted and actual values, while the confusion matrix provides insight into model accuracy, precision, recall, and F1-score in classification settings.  

### Confusion Matrix Explanation  

A confusion matrix is a performance evaluation tool for classification models, including decision trees. It summarizes the model’s predictions by comparing actual versus predicted values.  

For a binary classification problem (e.g., predicting whether an employee completes more than 75% of projects), the confusion matrix consists of four values:  

|                | **Predicted 0** | **Predicted 1** |  
|--------------|----------------|----------------|  
| **Actual 0**  | True Negatives (TN) | False Positives (FP) |  
| **Actual 1**  | False Negatives (FN) | True Positives (TP) |  

<scr>Definitions:<scr/>
- **Accuracy** = (TP + TN)/(TP + TN + FP + FN)) 
  - Measures overall correctness of the model.  
- **Precision** = (TP)/(TP + FP) 
  - Measures how many of the predicted positive cases were actually positive.  
- **Recall (Sensitivity)** = (TP)/(TP + FN)  
  - Measures how many actual positive cases were correctly predicted.  
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)  
  - Harmonic mean of precision and recall, balancing both metrics.  

The heatmap visualization provides a representation of these values, where darker colors indicate higher frequencies.  


## Clustering Method Explanation

K-Means Clustering is an unsupervised learning algorithm that groups data points into K clusters based on similarity. It works by:  

1. Initializing K centroids (randomly selecting K data points as initial cluster centers).  
2. Assigning each point to the nearest centroid based on Euclidean distance.  
3. Updating centroids** by calculating the mean position of all points in each cluster.  
4. Repeating steps 2 and 3** until centroids stabilize (i.e., do not change significantly).  

To evaluate clustering quality, the Silhouette Score** measures how well each point fits within its assigned cluster (ranging from 0 to 1, where a higher value indicates better-defined clusters).  


# Results Analysis

## Neural Network Results
We analyzed predictions from our neural network model that predicts employee job satisfaction based on business travel frequency, distance from home, years at company, and age. The model predicts satisfaction on a scale that appears to range from approximately 2.5 to 7.6.


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

### Pattern Analysis:

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

### Notable Relationships:

### Age-Distance Interaction
- Older employees with longer commutes show surprisingly high satisfaction
- This might suggest successful career advancement offsetting commute burden

### Travel-Tenure Relationship
- Frequent travel combined with long tenure predicts higher satisfaction
- May indicate career success and advancement opportunities

### Model Behavior Insights
1. **Range**: Predictions span from 2.51 to 7.56, showing significant variation
2. **Clustering**: Several predictions cluster around 2.6-2.8 for younger employees
3. **Outliers**: Notably high predictions for senior employees with long distances

### Inspirations for Further Investigation
1. Validate the high satisfaction predictions for long-distance senior employees
2. Investigate why younger employees consistently show lower satisfaction
3. Explore additional features that might explain satisfaction variations
4. Consider potential biases in the training data regarding age and tenure

### Technical Notes
- The model shows consistent patterns in prediction behavior
- TensorFlow warnings indicate potential optimization opportunities in the prediction pipeline
- Results suggest strong feature interactions affecting predictions

### Usage Example
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
## Cluster Graph Results

![image](https://github.com/user-attachments/assets/cde952e3-a7b8-484c-bfa5-cea11b8ab1fd)


### Overview
This visualization demonstrates k-means clustering analysis of employee data, plotting salary against the number of projects completed. The analysis reveals four distinct clusters, each marked by a red centroid (X), suggesting natural groupings in the workforce.

### Cluster Observations

### Entry-Level Cluster (Blue)
- Located in the bottom-left quadrant
- Salary range: ~30,000-50,000
- Projects completed: 0-7
- Characteristics: Likely represents entry-level or junior positions with limited project experience

### Mid-Level Cluster (Green)
- Located in the lower-middle section
- Salary range: ~55,000-75,000
- Projects completed: 5-15
- Characteristics: Represents mid-level employees with growing project experience

### Senior-Level Cluster (Purple)
- Located in the middle-upper section
- Salary range: ~75,000-95,000
- Projects completed: 8-20
- Characteristics: Indicates senior positions with substantial project experience

### Expert-Level Cluster (Yellow)
- Located in the top-right quadrant
- Salary range: ~95,000-120,000
- Projects completed: 15-25
- Characteristics: Represents expert-level positions with extensive project experience

### Key Insights
1. Clear positive correlation between projects completed and salary
2. Distinct salary bands with minimal overlap between clusters
3. Project experience appears to be a strong indicator of salary level
4. Natural progression paths visible through the cluster arrangement

### Technical Details
- Visualization Type: Scatter plot with k-means clustering
- Number of Clusters: 4
- Axes: X-axis shows Projects Completed (0-25), Y-axis shows Salary (30,000-120,000)
- Centroids marked with red X markers indicating cluster centers

![image](https://github.com/user-attachments/assets/33a67e74-6dfa-4a4b-afbe-25b755e36673)

### Overview
This visualization presents a k-means clustering analysis plotting employee satisfaction rates against the number of projects completed. The data is segmented into four distinct clusters, each with its own centroid (marked by red X), revealing different patterns in employee satisfaction levels.

### Cluster Observations:

### High Satisfaction Cluster (Blue)
- Located in the top portion of the graph
- Satisfaction range: 75-100%
- Projects completed: 0-25
- Characteristics: Represents consistently high employee satisfaction regardless of project count

### Medium-High Satisfaction Cluster (Yellow)
- Located in the upper-middle section
- Satisfaction range: 45-70%
- Projects completed: 0-25
- Characteristics: Shows moderately high satisfaction levels across various project counts

### Medium-Low Satisfaction Cluster (Green)
- Located in the lower-middle section
- Satisfaction range: 25-45%
- Projects completed: 0-25
- Characteristics: Indicates below-average satisfaction levels

### Low Satisfaction Cluster (Purple)
- Located in the bottom portion
- Satisfaction range: 0-25%
- Projects completed: 0-25
- Characteristics: Represents consistently low satisfaction rates

### Key Insights
1. No clear correlation between number of projects completed and satisfaction rates
2. Distinct satisfaction bands suggest systematic factors affecting employee experience
3. Wide spread of satisfaction levels across all project counts
4. Employee satisfaction appears to be independent of project volume

### Technical Details
- Visualization Type: Scatter plot with k-means clustering
- Number of Clusters: 4
- Axes: X-axis shows Projects Completed (0-25), Y-axis shows Satisfaction Rate (0-100%)
- Centroids marked with red X markers indicating cluster centers

 ![image](https://github.com/user-attachments/assets/4a30641b-247f-46fb-b35e-98c1d55327da)

# Satisfaction Rate vs Projects Completed: K-Means Clustering Analysis

## Overview
This visualization shows k-means clustering analysis of employee satisfaction rates plotted against the number of projects completed. The data is divided into four distinct clusters, each marked by a red centroid (X), revealing different patterns in employee satisfaction levels.

## Cluster Observations

### High Satisfaction Cluster (Yellow)
- Located in the top portion
- Satisfaction rate range: 25-40%
- Projects completed: 0-25
- Characteristics: Represents highest employee satisfaction spanning all experience levels

### Medium-High Satisfaction Cluster (Blue)
- Located in the upper-middle section
- Satisfaction rate range: 15-25%
- Projects completed: 10-25
- Characteristics: Shows moderate-high satisfaction among more experienced employees

### Medium-Low Satisfaction Cluster (Green)
- Located in the lower-middle section
- Satisfaction rate range: 5-12%
- Projects completed: 12-25
- Characteristics: Indicates decreasing satisfaction among highly experienced employees

### Low Satisfaction Cluster (Purple)
- Located in the bottom portion
- Satisfaction rate range: 0-15%
- Projects completed: 0-12
- Characteristics: Represents lower satisfaction rates among less experienced employees

## Key Insights
1. No clear positive correlation between project experience and satisfaction
2. Potential satisfaction issues with highly experienced employees (15+ projects)
3. Wide spread of satisfaction levels among newer employees (0-10 projects)
4. Notable cluster of consistently high satisfaction (25-40%) across all experience levels

## Technical Details
- Visualization Type: Scatter plot with k-means clustering
- Number of Clusters: 4
- Axes: X-axis shows Projects Completed (0-25), Y-axis shows Satisfaction Rate (0-40%)
- Centroids marked with red X markers indicating cluster centers

## What to do Next Time
1. Investigate factors contributing to low satisfaction in experienced employees
2. Study practices of consistently high-performing teams/departments
3. Develop targeted engagement strategies for different employee segments
4. Consider workload optimization for employees with high project counts

## Decision Tree Results
![image](https://github.com/user-attachments/assets/6da0c2d7-0331-4884-9392-54fee09c9ead)

### Overview
This decision tree visualization represents a model for predicting whether an employee will perform above the 75th percentile, based on various workplace factors. The tree uses multiple variables to classify high performers, with each node showing decision criteria and statistical metrics.

### Key Decision Paths:

### Primary Split
- Root node based on Salary (threshold: 100757.0)
- 180 total samples with squared error of 0.146
- Indicates salary as primary predictor of high performance

### Left Branch (Lower Salary)
1. Initial split on Productivity (threshold: 97.5%)
   - 133 samples, squared error 0.05
   - High performers tend to already show high productivity
   - Further refined by:
     - Secondary salary threshold (86759.0)
     - Satisfaction Rate (11.5%)
     - Distance From Home (8.5)
     - Percent Salary Hike (14.5%)

### Right Branch (Higher Salary)
1. Split on Position (threshold: 4.0)
   - 47 samples, higher error rate (0.249)
   - Further refined by:
     - Environment Satisfaction (2.5)
     - Total Working Years (19.0)
     - Satisfaction Rate (97%)

### Key Performance Indicators
1. Salary (primary determinant)
2. Current Productivity
3. Position Level
4. Environment Satisfaction
5. Total Working Years
6. Satisfaction Rate
7. Distance From Home
8. Percent Salary Hike

### Model Insights
1. High salary alone doesn't guarantee top performance
2. Current productivity is a strong predictor for lower-salaried employees 
3. Position level becomes more important for higher-salaried employees
4. Employee satisfaction metrics play significant roles in both branches
5. Work experience (Total Working Years) matters more for higher-salaried positions
6. Distance from home can impact performance in lower-salaried positions

### Statistical Notes
- Model uses squared error as measure of node purity
- Smaller nodes (1-4 samples) suggest possible overfitting
- More complex decision paths for lower-salaried employees
- Higher error rates in the right branch suggest more variability in high-salary performance prediction

This decision tree provides a framework for understanding the factors that contribute to exceptional employee performance, with clear differentiation between salary levels and their associated performance drivers.
# Model Evaluations

## Neural Network Evaluation

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

## Cluster Graph Evaluation:
The Elbow Method is a technique used in K-Means clustering to determine the optimal number of clusters by plotting the Sum of Squared Errors (SSE) for different values of 
𝑘 and identifying the point where the decrease in SSE slows down, forming an "elbow." Here, we can see that 4 clusters was best. 
![image](https://github.com/user-attachments/assets/63e85af3-f1ed-4f06-9ad5-9123a961a477)

Here are the silhouette scores for the graphs in the order in which they appear in the results section. The higher the value, the more clustered the graphs are.
```
Silhouette Score: 0.5878064086134991
```
```
Silhouette Score: -0.05405965558639975
```
```
Silhouette Score: 0.025078394835782555
```

## Decision Tree Evaluation

# Model Performance Analysis: Confusion Matrix and Metrics

![image](https://github.com/user-attachments/assets/f2cec888-481d-4c3f-809a-1f7e885d4dc6)

```
Accuracy: 0.8
Precision: 0.75
Recall: 0.5
F1 Score: 0.6
```
## Confusion Matrix Breakdown
- True Negatives (TN): 13 cases
- False Positives (FP): 1 case
- False Negatives (FN): 3 cases
- True Positives (TP): 3 cases
- Total Predictions: 20 cases

## Performance Metrics
- **Accuracy**: 0.80 (80%)
  - Model correctly predicts 16 out of 20 cases
  - Shows good overall performance

- **Precision**: 0.75 (75%)
  - Of all positive predictions, 75% were correct
  - Indicates good reliability when model predicts positive class

- **Recall**: 0.50 (50%)
  - Model correctly identifies 50% of all actual positive cases
  - Suggests room for improvement in identifying positive cases

- **F1 Score**: 0.60 (60%)
  - Harmonic mean of precision and recall
  - Indicates moderate balanced performance

## Strengths
1. High accuracy (80%) shows good overall performance
2. Strong precision (75%) indicates reliable positive predictions
3. Good at identifying negative cases (13 out of 14 correct)

## Areas for Improvement
1. Lower recall (50%) suggests model misses half of positive cases
2. False negatives (3) indicate potential missed opportunities
3. F1 score (60%) suggests room for better balance between precision and recall

## What to do next time
1. Consider rebalancing model to improve recall if missing positive cases is costly
2. Investigate patterns in false negatives to understand missed positive cases
3. Evaluate if current performance metrics align with business requirements
4. Consider collecting more training data for positive cases

This model shows strong overall accuracy but may benefit from optimization depending on the specific business needs and costs associated with false negatives versus false positives.

# Contributions

### **Cailey Crandall**  
- Designed and programmed the neural network
- Authored README sections:  
  - **Data Source**  
  - **Installation and Setup**  
  - **Neural Network Overview**  
  - **Data Processing of Neural Network**  
  - **Creation Process of Neural Network**  
  - **Explaining the Math**  
  - **Results Analysis**
  - **Model Evaluations** 
  - **Future Work**
- Formatted the README

### **Vance Dimmick**  
- Designed and programmed the clustering model  
- Designed and programmed the decision tree  
- Authored README sections:  
  - **Decision Tree Overview**  
  - **Clustering Model Overview**  
  - **Data Processing for Decision Tree**  
  - **Creation Process for Decision Tree**  
  - **Creation Process of Cluster Model**  

 ### Sheridan Traish
- Authored README sections:  
  - **Assumptions and Questions**  
### Renee Vannice 
- Authored README sections:  
  - **Introduction**  
  - **Data Source**  
  - **Prior Research**  
  - **Acknowledgements**
 
# Future Work
- Feature Engineering: Investigate additional features (e.g., demographic data, employee engagement survey results) to enhance model accuracy.
- Model Optimization: Hyperparameter tuning for improved performance across all models.
- Deployment: Packaging models for real-time use within an organization’s HR analytics pipeline.

# Acknowledgments
- https://www.forbes.com/sites/barnabylashbrooke/2023/07/13/job-satisfaction-is-key-to-workplace-productivity-but-how-do-you-get-it/
- https://www.indeed.com/career-advice/career-development/positive-working-environment





