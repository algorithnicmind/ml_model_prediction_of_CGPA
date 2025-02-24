# Prediction Using Machine Learning Algorithm

## Project Overview
This project predicts salary packages based on CGPA using Machine Learning. A simple linear regression model is used to analyze the relationship between CGPA and package, with further enhancements incorporating additional features like random values and IQ scores.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Algorithm**: Linear Regression
- **Data Processing & Visualization**: Pandas, NumPy, Matplotlib
- **Model Evaluation**: Sklearn Metrics

## Steps Involved
1. **Import Required Libraries**  
   - pandas, numpy, matplotlib.pyplot, sklearn (train_test_split, LinearRegression, metrics)
2. **Load the Dataset**  
   - Read the dataset (CSV format) using Pandas
3. **Visualize Data**  
   - Scatter plot for CGPA vs Package
4. **Data Preparation**  
   - Extract features (X) and target variable (y)
   - Split data into training and testing sets
5. **Train the Model**  
   - Fit a Linear Regression model on training data
6. **Make Predictions**  
   - Predict package values for test data
7. **Evaluate the Model**  
   - Calculate MAE, MSE, RMSE, and R² score
   - Compute adjusted R² score
8. **Feature Expansion**  
   - Add a random feature and analyze its impact
   - Introduce an IQ-based feature and re-train the model
9. **Final Model Evaluation**  
   - Compute R² score with new features
   - Calculate model parameters (slope & intercept)

## Libraries Used
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Scikit-learn**:
  - `train_test_split`: Splitting dataset into training/testing
  - `LinearRegression`: Applying Linear Regression
  - `mean_absolute_error`, `mean_squared_error`, `r2_score`: Model evaluation metrics

## Results
- Initial linear regression provided basic package predictions.
- Adding additional features like a random variable and IQ scores tested model robustness.
- R² and adjusted R² scores were computed to assess the model's performance.

## How to Run
1. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Place `placement.csv` in the project directory.
3. Run the script using:
   ```bash
   python script.py
   ```
4. View scatter plots and model performance metrics.

## Conclusion
This project demonstrates the impact of linear regression in predicting package values based on CGPA, explores additional features, and evaluates model performance using key metrics.

---
**Author:** Ankit Sahoo

if you did not understand this you can conatant me 

