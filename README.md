# Flight-Delay-Analysis-And-Cross-Validation

In this project, I am performing the following tasks:

Part I:

- Importing necessary libraries: numpy, scipy, pandas, matplotlib, sklearn, and statistics.
- Reading a flights dataset from a CSV file using pandas: flights_df = pd.read_csv('flights.csv').
- Performing various operations on the flights dataset, such as checking its shape, column names, data types, unique destinations, and displaying the top rows.
- Answering specific questions related to the flights dataset, such as the number of flights from NYC to Seattle, the number of airlines flying from NYC to Seattle, the number of unique airplanes flying from NYC to Seattle, the average arrival delay for flights from NYC to Seattle, and the proportion of flights to Seattle from each NYC airport.

Part II:

- Importing necessary libraries: seaborn, LinearRegression from sklearn.linear_model, KNeighborsRegressor from sklearn.neighbors, preprocessing from sklearn, KFold and cross_validate from sklearn.model_selection, r2_score and mean_squared_error from sklearn.metrics.
- Reading a ship dataset from a CSV file using pandas.
- Exploring the ship dataset.
- Preparing the data for cross-validation by selecting features (X) and the target variable (y) and defining a KFold cross-validation split.
- Training a linear regression model and a KNN model (k=3) using cross-validation and evaluating their performance in terms of R-squared (R2) score and mean squared error (MSE).
- Performing a shortcut to do cross-validation using the cross_validate function from scikit-learn and calculating the average R2 score and MSE for both models.
- Performing a hyperparameter tuning for the KNN model by varying the number of neighbors (k) from 1 to 19 and evaluating the training and test errors.
- Plotting the training and test errors for different values of k.
- Performing feature scaling on the ship dataset using standardization (StandardScaler) and training the linear regression and KNN models with the scaled features.
- Evaluating the performance of the scaled models and comparing them with the unscaled models.
