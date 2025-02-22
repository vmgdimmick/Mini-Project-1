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
Our data is derived from two separate Kaggle datasets, cleaned and merged into a single comprehensive dataset for the clustering model.




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



