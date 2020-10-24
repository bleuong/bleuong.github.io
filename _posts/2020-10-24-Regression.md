---
title:  "Regression"
date:   2020-10-24 14:33:00 +0800
categories: Regression
tags: linear polynomial regression support_vector decision_tree random_forest r-squared
---

# Regression

It has been a month since I had last updated this blog. During this time, I am taking a Data Science course from Udemy: [Machine Learning A-Z Hands On Python and R in Data Science]. This course is by Kirill Eremenko and Hadelin de Ponteves. To be more precise, I am re-taking the course. I had bought the course in late 2019 and started a bit on regression and classification. Due to work committments, I stopped for a while before resuming now.

I am glad I had restarted the learning. In tandem with the fast paced and continuous development in the world of Data Science, the course had been updated also. The main difference was the practical lessons. When I first started, programming lessons was conducted using Spyder. Since Jun 2020, the practical lessons had been updated to use Goodle Colaboratory in the cloud. The lessons content were also enhanced based on the questions and feedback.

Another reason I liked the course is the simplicity of the explanation behind every algorithm. The instructors refrained from going into mathematics and tried using examples to simplify the intuition behind the algorithm. For my purpose of learning to be a data analyst, it helps tremendously.

Enough of the batter about this course. Let's get into today's sharing.

## Key Learning Points

Regression is simply to fit a model to predict a numerical outcome, e.g. sales volume, temperature, or price. Since the model is an approximation to the actual data, there will be deviations between the actual data and model prediction. Every model will seek to minimise this deviation in different ways. In this course, the following regression models are covered.

1. Multiple Linear Regression
2. Polynomial Linear Regression
3. Support Vector Regression
4. Decision Tree Regression
5. Random Forest Regression

Different models will come up with different prediction results. The classical way is to run the data through all the models with their baseline parameters and determine which model comes up with the best predictions. In this case, the R-Squared value for each model is used as the comparison metric. After the model with the highest R-Sqaured value is selected, fine tuning of the parameters can be done to improve the results.

The code to be shared would run through all the method on the data and provide the R-Squared scores. Parameter tuning is not covered yet as that is the last part of the Udemy course which I had not covered. So i will update parameter tuning at a later time.

The source code and sample data file can be found [here].

### Data Preprocessing

This is the same data preprocessing template as before. So no surprises.


```python
# Import required libraries for the entire code
import numpy as np  # library for numerical manimuplation
import matplotlib.pyplot as plt  # library for plotting
import pandas as pd  # library for importing datasets


def data_preprocessing(input_file_name=None,
                       imputer_input={"selected_col": None,
                                      "strategy": "mean"},
                       x_encoder_input={"cols": None, "AvoidDummy": False},
                       y_encoder_input=False):
    """
    This is to import the dataset from the given csv file and process it for
    subsequent analysis. The data file must be located in the same folder as
    this python script. The processing includes 1) Data import, 2) Fill in
    missing values, and 3) Encode categorical data.

    When encoding categorical data, the new variables created based on the
    categorical values are called dummy variables. It is important to avoid
    using all the dummy variables (dummy variable trap) as the model
    would be unable to differentiate the relationship. Hence, always use 1 less
    dummy variable in the model for each set of categorical variables.

    It is important to arrange the dataset columns in the following sequence
    from left to right.
    1. Numrical data (for filling in of missing values)
    2. Categorical data (after encoding, it would be the first few columns in
                          the dataset. Remove the first column to avoid dummy
                          variable trap before encoding the next categorical
                          variable.)
    3. Dependent variable

    Inputs
    ------
    input_file_name : This is the name of the file to be imported for analysis.
                        The file must be located in the same folder as this
                        python script.
    imputer_input : A dict containing a list of columns to fill in the missing
                    values and the strategy for the missing values.
                    Can be "mean", "median", "most_frequent". If the list
                    of columns is None, it means no missing values.
    x_encoder_input : A dict containing a list of columns index in x to encode
                        the categorical data and a boolen value to indicate if
                        there is a need to avoid the dummy variable trap. An
                        empty list indicate no need to perform encoding.
                        Default is {cols: [], AvoidDummy=False}
    y_encoder_input : A boolen value indicating if the dependent variable y is
                    categorical data and need to be encoded. Default is False.

    Parameters
    ----------
    dataset : Variable containing the full data after importing from csv file.
    imputer : SimpleImputer object created to fill in missing values
    ct : ColumnTransformer object created to encode the categorical data in x

    Returns
    -------
    x : Variable containing all independent features. Numpy array type
    y : Variable containing all dependent value. Numpy array type

    Pseudo Code
    -----------
    Import data
        Import from csv file
        Extract to x and y and return them
    Take care of missing data
        If missing data is less than 1% of the total data, then can just
        delete them away.
        If there are columns with missing data
            Replace with mean/median/most_frequent of existing data.
    Encode categorical data
        If there are columns in x with categorical data:
            Use OneHotEncoder to encode each respective columns
            OneHotEncoder will encode and transfer the dummy variables to the
            first column of the dataset.
        If the dependent variable y is categorical:
            Use LabelEncoder to encode
    Return x, y
    """

    """ Import data """

    # Read data from data file
    dataset = pd.read_csv(input_file_name)
    print("Data imported successfully.\n")

    print("dataset Info")
    print(dataset.info(), "\n")
    print("dataset (First 5 rows)")
    print(dataset.head(), "\n")

    # Extract the dependent variable (y) and independent features (x).
    # It is assumed the dependent variable column is the last one.
    # All rows are selected. All columns except last one is selected.
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Reshape y into a single column
    y = y.reshape(-1, 1)

    """ Take care of missing data """
    # Check if there is missing data to be filled in
    if imputer_input["selected_col"] is not None:
        # User has provided columns index which has missing data
        from sklearn.impute import SimpleImputer

        # Create the imputer object to be used on x, y
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy=imputer_input["strategy"])

        # Cycle through every column index provided by user
        for col_index in imputer_input["selected_col"]:
            # Link the imputer object to the numerical columns in x with the
            # correct column index
            # Need to reshape (-1, 1) each column for the imputer object to
            # work and reshape it back (1, -1) to put back into x.
            imputer.fit(x[:, col_index].reshape(-1, 1))
            # Perform the actual replacement on x
            x[:, col_index] = \
                imputer.transform(x[:, col_index]
                                  .reshape(-1, 1)).reshape(1, -1)

        print("Missing data processed.\n")
    else:
        # The selected_col list is None. Hence no missing data.
        print("Process missing data skipped. \n")

    """ Encode categorical data """
    # Check if there is categorical data in x to encode
    if x_encoder_input["cols"] is not None:
        # Column index had been provided by user to encode
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        # Cycle through each selected column index for encoding
        for selected_col in x_encoder_input["cols"]:
            # Create ColumnTransformer object
            ct = ColumnTransformer(transformers=[("encoder",
                                                  OneHotEncoder(),
                                                  [selected_col])],
                                   remainder="passthrough")
            # Fit and transform the ct object to x in one step
            # Need to use np.array to force fit the result from fit_transform
            # into a numpy array for subsequent analysis
            x = np.array(ct.fit_transform(x))

            # Remove first column of dummy variable to avoid dummy variable
            # trap. The dummy variables are arranged in alphabetical order.
            if x_encoder_input["AvoidDummy"]:
                # Need to avoid dummy variable trap because the model is unable
                # to take care of the dummy variable
                x = x[:, 1:]
            else:
                # Model is able to avoid dummy variable trap. Hence no need to
                # remove any dummy variable.
                pass

        print("Categorical data encoded.\n")
    else:
        # User did not provide any columns for encoding. So do nothing.
        print("Categorical data processing skipped.")

    # Check if need to encode dependent variable y
    if y_encoder_input:
        # Dependent variable y is categorical and need to be encoded
        from sklearn.preprocessing import LabelEncoder

        # Create LabelEncoder object
        le = LabelEncoder()
        # Fit_transfer LabelEncoder onto dependent variable y
        y = le.fit_transform(y)

        print("Categorical data in y is encoded.\n")
    else:
        # Dependent variable y is not categorical.
        pass

    return x, y


def split_dataset(x, y=None, split_data_input=[0.2, None]):
    """
    This is to split the dataset into the training set and test set.

    Inputs
    ------
    x : numpy array of independent features
    y : a list of dependent variable. Default is None
    split_data_input : A list containing 2 inputs required to split the
                        dataset. [test_size=0.2, random_state=None]

    Parameters
    ----------
    None

    Returns
    -------
    x_train, y_train : Training set for x and y
    x_test, y_test : Test set for x and y

    Pseudo Code
    -----------
    Use train_test_split from sklearn.model_selection
    Return x_train, y_train, x_test, y_test
    """

    from sklearn.model_selection import train_test_split

    x_train, x_test = \
        train_test_split(x, test_size=split_data_input[0],
                         random_state=split_data_input[1])

    if y is not None:
        # y is given and need to be split
        y_train, y_test = \
            train_test_split(y, test_size=split_data_input[0],
                             random_state=split_data_input[1])
    else:
        # y is not given
        y_train, y_test = None, None

    # Check if x_train and x_test are 2 dimensional array. Reshape if necessary
    if len(x_train.shape) == 1:
        # 1 dimensional list. Reshape
        x_train = x_train.reshape(len(x_train), 1)
        x_test = x_test.reshape(len(x_test), 1)
    else:
        # x_train is 2 dimensional array. Do nothing
        pass

    return x_train, y_train, x_test, y_test


def Features_Scaling(x_train, y_train=None, scaler_input=None,
                     x_test=None, y_test=None):
    """
    This function is to scale the features if required on the independent
    variables, x. This is highly dependent on the selected model. Will be
    using the standardisation method which is applicable to all data instead
    of the normalisation method which can only be used on data with normal
    distribution.

    Inputs
    ------
    x_train, y_train : Training set for x.
    y_train : Training set for y. Assumed to be a 1 col array. Default is None.
    scaler_input : A list containing the columns to be scaled. Default is None,
                    which means scale all features. Only applicable for x_train
    x_test, y_test : Test set for x and y which required to be scaled. Default
                        is None. y_test is assumed to be a 1 column array.

    Parameters
    ----------
    sc_x_train, sc_y_train : StandardScaler object created to transform the
                                training set.

    Returns
    -------
    x_train_sc, y_train_sc, sc_x_train, sc_y_train, x_test_sc, y_test_sc

    Pseudo Code
    -----------
    Create StandardScaler object to fit and transform x_train into x_train_sc
    If scaler_input is provided, replace columns in x_train_sc by corresponding
    columns in x_train if the columns index is not specified in scaler_input
    else do nothing since the entire x_train had already been transformed.

    If y_train is provided:
        If y_train is not 1D array, create StandardScaler object to fit and
        transform y_train into y_train_sc
    else set y_train_sc and sc_y_train to None

    If x_test is provided, use sc_x_train to scale x_test into x_test_sc.
        If scaler_input is provided, replace columns in x_test_sc by
            corresponding columns in x_test if the columns index is not
            specified in scaler_input.
    else set x_test_sc to None

    If y_test is provided, use sc_y_train to scale y_test into y_test_sc.
    else set _test_sc to None

    return x_train_sc, y_train_sc, sc_x_train, sc_y_train, x_test_sc, y_test_sc
    """

    from sklearn.preprocessing import StandardScaler

    # Create standard scaler object to be used on x_train
    sc_x_train = StandardScaler()
    x_train_sc = sc_x_train.fit_transform(x_train)

    # if scaler_input is provided, only specific columns to be scaled.
    # Replace columns in x_train_sc with x_train which need not be scaled
    if scaler_input is not None:
        for i in range(x_train.shape[1]):
            # Cycle through every column in x_train
            if i not in scaler_input:
                x_train_sc[:, i] = x_train[:, i]
            else:
                # ith column needs to be scaled, so do nothing
                pass
    else:
        # scaler_input not provided. So scale the entire x_train. Do nothing.
        pass

    """ Scale y_train if provided """
    if y_train is not None:
        # Create StandardScaler object and fit onto y_train
        if len(y_train.shape) == 1:
            # y_train is 1D array. Reshape into 2D array.
            y_train = y_train.reshape(len(y_train), 1)
        else:
            # y_train is not 1D array. Do nothing
            pass

        sc_y_train = StandardScaler()
        y_train_sc = sc_y_train.fit_transform(y_train)
    else:
        # y_train not provided
        y_train_sc, sc_y_train = None, None

    """ Scale x_test if provided """
    if x_test is not None:
        # Scale using sc_x_train
        x_test_sc = sc_x_train.transform(x_test)

        # if scaler_input is provided, only specific columns to be scaled.
        # Replace columns in x_test_sc with x_test which need not be scaled
        if scaler_input is not None:
            for i in range(x_test.shape[1]):
                # Cycle through every column in x_test
                if i not in scaler_input:
                    x_test_sc[:, i] = x_test[:, i]
                else:
                    # ith column need to be scaled. Do nothing
                    pass
        else:
            # scaler_input not provided. So scale the entire x_test. Do nothing
            pass
    else:
        # x_test not provided
        x_test_sc = None

    """ Scale y_test if provided """
    if y_test is not None:
        # y_test is provided
        if len(y_test.shape) == 1:
            # y_test is 1D array. Reshape to 2D array
            y_test = y_test.reshape(len(y_test), 1)
        else:
            # y_test is not 1D array. Do nothing
            pass

        # Use sc_y_train to scale y_test
        y_test_sc = sc_y_train.transform(y_test)
    else:
        # y_test is not provided.
        y_test_sc = None

    return x_train_sc, y_train_sc, sc_x_train, sc_y_train,\
        x_test_sc, y_test_sc

```

### Main Regression Module

This is the main regression module which contains the code to perform Multiple Linear Regression, Decision Tree Regression, and Random Forest Regression. It uses the Scikit-Learn module as the basis.


```python
def Regression(x_train, y_train, resolution=None, reg_model="linear",
               x_test=None, y_test=None, x_pred=None,
               x_train_plot_poly=None, ntree=0):
    """
    This function uses multiple linear regression, Decision Tree Regression to
    fit the data. It can be used for simple linear regression too.

    In skelearn, the LinearRegression library would account for dummy variable
    trap and select the variables with the highest p-value. Hence there is no
    need to select variables to build the model.

    Regression does not require feature scaling because the constant for each
    variable would take care of the scaling.

    Inputs
    ------
    x_train, y_train : Training set for x and y.
    resolution : No of points for a high resolution plot of the regression
                    line. Default is 100. Applicable only to one dimensional x.
    reg_model : Indicate the regression model to be used which includes linear,
                decisiontree, randomforest. Default is linear.
    x_test, y_test : Test set for x and y. Default is None if no test set is
                        available
    x_pred : A list containing a single sample of x for prediction
    x_train_plot_poly : High resolution training set for x specifically for
                        polynomial model
    ntree : Number of trees to be used for Random Forest Regression model.
            Default is 0.

    Parameters
    ----------
    model : model object to fit x_train and y_train
    x_train_plot : a high resolution list of x_train for plotting regression
                    line.
    y_train_reg : predicted results for dependent variable y based on x_train
    y_test_pred : Predicted results for test data using trained model on
                    x_test
    y_pred : A single value prediction using x_pred
    r2 : R2 score using y_test and y_test_pred. Used to assess if model is
            suitable for the dataset

    Returns
    -------
    x_train_plot, y_train_reg, y_test_pred, y_pred, r2

    Pseudo Code
    -----------
    Create the model object and fit to x_train, y_train
    Print intercept and coefficients
    If x is 1 dimensional, define high resolution x_train_plot and get high
        resolution regression line
    Else get regression line using x_train or x_train_plot_poly (for polynomial
                                                                 model)
    if x_test is provided, predict y_test_pred to compare with y_test
    Predict for a single value x_pred if it is given
    Return x_train_plot, y_train_reg, y_test_pred, y_pred
    """

    from sklearn.metrics import r2_score

    # Create model obejct based on required model
    if reg_model == "linear":
        # Linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif reg_model == "decisiontree":
        # Decision Tree Regression
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
    elif reg_model == "randomforest":
        # Random Forest Regression
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=ntree)
    else:
        # Invalid input
        raise ValueError("The selected regression model is invalid.")

    # Fit the data to the model object (training the model)
    model.fit(x_train, y_train)

    # To get the coeeficients for the simple linear regression line
    # if reg_model == "linear":
    #     print(f"Intercept b0 = {float(model.intercept_):0.1f}")
    #     print(f"Coefficient = {model.coef_}")
    # else:
    #     # Not linear regression. Do nothing
    #     pass

    # if x_train is one column and resolution for curve is provided, can do
    # high resolution regression line for y_train_pred
    if x_train.shape[1] == 1 and resolution is not None:
        # Only 1 dimension for x_train. Hence can do high resolution plot
        x_train_plot = np.arange(min(x_train), max(x_train),
                                 step=(max(x_train)-min(x_train))/resolution)
        x_train_plot = x_train_plot.reshape(len(x_train_plot), 1)
        y_train_reg = model.predict(x_train_plot)
    else:
        # x_train has multiple features. Plot regression line using original
        x_train_plot = x_train[:]
        if x_train_plot_poly is None:
            # Model is not polynomial. Use original x_train for regression line
            y_train_reg = model.predict(x_train)
        else:
            # Model is polynomial. Use x_train_plot_poly for regression line
            y_train_reg = model.predict(x_train_plot_poly)

    # Predict the test set results if test data is available
    if x_test is not None:
        # Test data available. Predict y_test_pred using x_test. This can be
        # compared with y_test
        y_test_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_test_pred)
    else:
        # Test data not available.
        y_test_pred, r2 = None, None

    # To predict a value: model.predict([[value]])
    # Take note must always put in [[]] because it expects a 2D array
    # For Multiple Linear Regression, the inputs will be put as a list inside
    # [[b0, b1, b2, b3...bn]]
    if x_pred is not None:
        # If x_pred is given, compute y_pred
        y_pred = model.predict([x_pred])
    else:
        # No x_pred given, so set y_pred to None
        y_pred = None

    return x_train_plot, y_train_reg, y_test_pred, y_pred, r2

```

### Polynomial Regression

This module is used to transform the data into a polynomial format before passing to the main regression module using the multiple linear regression method. 


```python
def polynomial_regression(x_train, resolution=100,
                          x_test=None, x_pred=None, n=2):
    """
    This function prepares all the independent vairables x matrixes into the
    polynomial form for fitting the Linear Regression model.

    Inputs
    ------
    x_train : Training set for x
    resolution : Number of points to plot for a high resolution regression line
    x_test : Test set for x
    x_pred : A list containing the dependent values for prediction
    n : required degree of power for the polynomial regression model. Default
        value is 2

    Parameters
    ----------
    poly_reg : Polynomial model of degree n
    x_train_plot : A numpy array holding high resolution points for x_train.

    Returns
    -------
    x_train_poly, x_test_poly, x_pred_poly : Training and test set of dependent
                                                variable x after being
                                                transformed into polynomial

    Pseudo Code
    -----------
    Create polynomial model
    Use polynomial model on x_train, x_test and x_pred
    """

    from sklearn.preprocessing import PolynomialFeatures

    # Create the polynomial model to transform x_train, x_test
    poly_reg = PolynomialFeatures(degree=n)
    x_train_poly = poly_reg.fit_transform(x_train)

    # Define high resolution plot for x_train
    if resolution is not None:
        # Resolution is given. Hence can plot.
        x_train_plot = np.arange(min(x_train), max(x_train),
                                 step=(max(x_train)-min(x_train))/resolution)
        x_train_plot = x_train_plot.reshape(len(x_train_plot), 1)
        x_train_plot_poly = poly_reg.fit_transform(x_train_plot)
    else:
        # No need to plot
        x_train_plot, x_train_plot_poly = None, None

    if x_test is not None:
        # x_test is provided. Transform x_test
        x_test_poly = poly_reg.fit_transform(x_test)
    else:
        # x_test is not provided
        x_test_poly = None

    if x_pred is not None:
        # x_pred is provided. Transform x_pred
        [x_pred_poly] = poly_reg.fit_transform([x_pred])
    else:
        # x_pred not provided
        x_pred_poly = None

    return x_train_poly, x_train_plot, x_train_plot_poly, \
        x_test_poly, x_pred_poly

```

### Support Vector Regression

This module is used primarily for support vector regression.


```python
def support_vector_regression(x_train, y_train, sc_x_train, sc_y_train,
                              resolution=None,
                              x_test=None, y_test=None, x_pred=None):
    """
    This function is to perform Support Vector Regression on the dataset. Do
    note that feature scaling is required for the dataset as the library is
    unable to take care of it inherently.

    This implementation uses the Gaussian Radial Basis Function Kernal.
    Refer to link in line 6 for to read up on other kernels which can be used.

    Inputs
    ------
    x_train, y_train : Training set for x and y. It must be scaled.
    sc_x_train, sc_y_train : Standard Scaler object used to scale x_train and
                                y_train.
    resolution : The number of points to be used for high resolution line.
                    Default is 100 points.
    x_test, y_test : Test set for x and y. It must be scaled.
                        Default is None if no test set is available
    x_pred : A list containing the independent values for prediction. Default
                is None.

    Parameters
    ----------
    model : SVR regression object
    x_train_plot : A list of x points for plotting the regression line.
    y_train_reg : Predicted values using x_train.
    y_test_pred : Predicted values using x_test. It must be scaled.
                    Default is None if no test set is available
    y_pred : A single prediction based on x_test
    r2 : R2 score for the model

    Returns
    -------
    x_train_plot, y_train_reg, y_test_pred, y_pred, r2

    Pseudo Code
    -----------
    Create SVR model and fit training data to it
    Create high resolution plot if there is only 1 feature
    Calculate regression values for y_train_reg using x_train_plot
    If x_test is provided, calculate prediction for y_test_pred
    If x_pred is provided, calculate prediction for y_pred
    return x_train_plot, y_train_reg, y_test_pred, y_pred
    """

    from sklearn.svm import SVR
    from sklearn.metrics import r2_score

    # Create the model for training set and fit it
    model = SVR(kernel="rbf")
    model.fit(x_train, y_train)

    # Create high resolution plot if x_train has only 1 feature and resolution
    # for the curve is provided
    if x_train.shape[1] == 1 and resolution is not None:
        # x_train only has one feature. Can create high resolution plot
        x_train_plot = np.arange(min(x_train), max(x_train),
                                 step=(max(x_train)-min(x_train))/resolution)
        x_train_plot = x_train_plot.reshape(len(x_train_plot), 1)
    else:
        # x_train has more than one feature. Set x_train_plot to x_train
        x_train_plot = x_train[:]

    # Predict the regression values and reverse the scaling on y_train_reg
    y_train_reg = sc_y_train.inverse_transform(model.predict(x_train_plot))
    # Reverse the scaling on x_train_plot
    x_train_plot = sc_x_train.inverse_transform(x_train_plot)

    # Create model for test set and fit it, if available
    if x_test is not None and y_test is not None:
        # x_test and y_test are provided. Predict base on x_test.
        y_test_pred = sc_y_train.inverse_transform(model.predict(x_test))
        r2 = r2_score(sc_y_train.inverse_transform(y_test), y_test_pred)
    else:
        # x_test and y_test not provided.
        y_test_pred, r2 = None, None

    if x_pred is not None:
        # Perform prediction. Remember to scale x_pred before the prediction
        # and reverse the scaling for y_pred.
        [y_pred] = sc_y_train.inverse_transform(model.predict
                                                (sc_x_train.transform
                                                 ([x_pred])))
    else:
        # x_pred not given. Return None
        y_pred = None

    return x_train_plot, y_train_reg, y_test_pred, y_pred, r2

```

### Plotting

This module can be used for plotting high resolution regression lines if applicable. It is not activated in the main template for regression model selection. But the code to activate is embedded in the main regression template as comments.


```python
def plot_regression(x_train, y_train, x_train_plot, y_train_reg,
                    plot_title, plot_xlabel, plot_ylabel,
                    x_test=None, y_test=None, y_test_pred=None):
    """
    This function plots the modeling results in 2 dimensional plot for
    regression models. All the inputs must be of the original form without any
    feature scaling.

    Inputs
    ------
    x_train, y_train : Training set for x and y
    x_train_plot, y_train_reg : High resolution regression line
    plot_title : title for the plot
    plot_xlabel : x-axis label for the plot
    plot_ylabel : y-axis label for the plot
    x_test, y_test : Test set for x and y if available. Default is None.
    y_test_pred : Predicted values of x_test using the model model

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    Pseudo Code
    -----------
    if x_train has more than one column:
        setup x_train as the case ID in x_train starting from 0
        setup x_train_reg as the case ID in x_train_reg starting from 0 but
            confined within the max of x_train.
    plot x_train vs y_train in red markers
    plot x_train_reg vs regression line (y_train_reg) in red line (high res)

    if x_test and y_test are provided:
        if x_test has more than one column:
            setup x_test as the case ID in x_test starting from 0
        plot x_test vs y_test in green markers
        if y_test_reg is provided:
            plot x_test vs y_test_reg in blue markers

    Set up plot title, axis labels and legends
    Show plot
    """

    # Visualise the training set results in a 2D plot
    if x_train.shape[1] > 1:
        # x_train has more than 1 feature. Set up case ID
        x_train = list(i for i in range(len(x_train)))
        x_train_plot = list(i/(len(x_train_plot)-1)*(len(x_train)-1)
                            for i in range(len(x_train_plot)))
    else:
        # x_train only has 1 column. No change required
        pass

    # Plot training data as red color scatter plots
    plt.scatter(x_train, y_train, label="Training Set Actual", color="red")

    # Plot high res regression line in red based on training data
    plt.plot(x_train_plot, y_train_reg, label="Training Set Regression",
             color="red", linestyle="-", marker="")

    # Visualise test set data in 2D plot if provided.
    if x_test is not None and y_test is not None:
        # Test data provided. Check number of features in x_test
        if x_test.shape[1] > 1:
            # x_test has more than 1 feature. Set up case ID
            x_test = list(i/(len(x_test)-1)*(len(x_train)-1)
                          for i in range(len(x_test)))
        else:
            # x_test has only 1 column. Do nothing
            pass

        # Plot x_test vs y_test in green markers
        plt.scatter(x_test, y_test, label="Test Set Actual", color="green")

        # Plot x_test vs y_test_reg in blue cross markers if available
        if y_test_pred is not None:
            plt.scatter(x_test, y_test_pred, label="Test Set Predicted",
                        color="blue", marker="x")
        else:
            # y_test_reg is not provided. Do nothing.
            pass

    # Set title
    plt.title(plot_title)
    # Set axis labels
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    # Display legend
    plt.legend(loc="best")
    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # Display the graph
    plt.show()

```

### Regression Model Selection

This is the main template to cycle through all the five regression models and find the best fitting model using the R-Squared value.


```python

def regression_model_selection():
    """
    This will be the main template to be used for regression. It will go
    through all the regression models and compare the R2 score using all the
    variables in the dataset. In order to select the best model, it is best to
    go through all the models. The comparison will be using the default
    parameters for each model. Tuning of the model is excluded from this module
    for the time being.

    Returns
    -------
    R2 scores for all the models for comparison.

    """

    # Data file name. File must be in same folder as script
    dataset = "RegressionData.csv"

    # Missing numerical data to be filled up
    # Default values "selected_col": [],"strategy": "mean"
    imputer_input = {"selected_col": [], "strategy": "mean"}

    # Columns in x with categorical data to encode
    # default {"cols": [], "AvoidDummy": False}
    x_encoder_input = {"cols": [], "AvoidDummy": False}
    # Indicate if need to encode dependent variable y, enter True or False
    y_encoder_input = False

    x, y = data_preprocessing(dataset, imputer_input, x_encoder_input,
                              y_encoder_input)
    print(f"Shape of x: {x.shape}")
    print(f"Shape of y: {y.shape}")

    # Inputs for splitting the dataset
    # State the random_state as 1 for troubleshooting. Use None for random.
    split_data_input = [0.2, None]

    # Always split dataset first prior to Feature scaling because you do not
    # want the mean value for the test set to be influenced by the training set
    x_train, y_train, x_test, y_test = split_dataset(x, y, split_data_input)

    # Set up the columns which needed to be scaled
    # scaler_input = []
    # Execute the scaling
    x_train_sc, y_train_sc, sc_x_train, sc_y_train, x_test_sc, y_test_sc = \
        Features_Scaling(x_train, y_train, None, x_test, y_test)

    # Title and labels for the plot
    # plot_title = "Position vs Salaries"
    # plot_xlabel = "Position Level"
    # plot_ylabel = "Salaries"

    # Initialise dependent value for prediction. Default is None.
    # The scaling will be performed in SVR module.
    # x_pred = []

    # Set number of points used for high resolution plot
    # Default is None
    resolution = None

    """Multiple Linear Regression"""
    reg_model = "linear"

    # Execute module
    x_train_plot, y_train_reg, y_test_pred, y_pred, linear_r2 = \
        Regression(x_train, y_train, resolution, reg_model, x_test, y_test)

    # print(f"{reg_model.title()} Regression Predicted "
    #       "Value:${float(y_pred)}")
    print(f"{reg_model.title()} R2: {linear_r2:0.3f}")

    """Polynomial Linear Regression"""
    # Initialise the expected polynomial degree
    n = 4

    # Select regression model
    reg_model = "linear"

    # Transform dependent variables x into polynomial format as input to Linear
    # Regression.
    x_train_poly, x_train_plot, x_train_plot_poly, x_test_poly, x_pred_poly = \
        polynomial_regression(x_train, resolution, x_test, None, n)

    # Execute Linear Regression on the data
    x_train_plot1, y_train_reg, y_test_pred, y_pred, polynomial_r2 = \
        Regression(x_train_poly, y_train, resolution, reg_model,
                   x_test_poly, y_test, None, None)

    # Plot results
    # plot_regression(x, y, x_train_plot, y_train_reg,
    #                 plot_title, plot_xlabel, plot_ylabel,
    #                 None, None, None)

    # Print y_pred if x_pred is given
    # print(f"For x = {x_pred}, the predicted value is ${float(y_pred):.2f}.")
    print(f"Polynomial Regression R2: {polynomial_r2:0.3f}")

    """Decision Tree Regression"""
    reg_model = "decisiontree"

    # Execute module
    x_train_plot, y_train_reg, y_test_pred, y_pred, decisiontree_r2 = \
        Regression(x_train, y_train, resolution, reg_model, x_test, y_test)

    # print(f"{reg_model.title()} Regression Predicted "
    #       "Value:${float(y_pred)}")
    print(f"{reg_model.title()} R2: {decisiontree_r2:0.3f}")

    """Random Forest Regression"""
    reg_model = "randomforest"
    # Set the number of trees if the regression model is Random Forest
    ntree = 200

    # Reshape y to row vector
    y_train_row = y_train.ravel()
    y_test_row = y_test.ravel()

    # Execute module
    x_train_plot, y_train_reg, y_test_pred, y_pred, randomforest_r2 = \
        Regression(x_train, y_train_row, resolution, reg_model,
                   x_test, y_test_row, ntree=ntree)

    # print(f"{reg_model.title()} Regression Predicted "
    #       "Value:${float(y_pred)}")
    print(f"{reg_model.title()} R2: {randomforest_r2:0.3f}")

    """Support Vector Regression"""
    # Reshape y to row vector
    y_train_sc = y_train_sc.ravel()
    y_test_sc = y_test_sc.ravel()

    x_train_plot, y_train_reg, y_test_pred, y_pred, svr_r2 = \
        support_vector_regression(x_train_sc, y_train_sc,
                                  sc_x_train, sc_y_train,
                                  resolution, x_test_sc, y_test_sc)

    # print(f"{reg_model.title()} Regression Predicted "
    #       "Value:${float(y_pred)}")
    print(f"Support Vector Regression R2: {svr_r2:0.3f}")

    # plot_regression(x, y, x_train_plot, y_train_reg,
    #                 plot_title, plot_xlabel, plot_ylabel)

    return linear_r2, polynomial_r2, decisiontree_r2, randomforest_r2, svr_r2

```

Activate the model.


```python
regression_model_selection()
```

    Data imported successfully.
    
    dataset Info
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9568 entries, 0 to 9567
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   AT      9568 non-null   float64
     1   V       9568 non-null   float64
     2   AP      9568 non-null   float64
     3   RH      9568 non-null   float64
     4   PE      9568 non-null   float64
    dtypes: float64(5)
    memory usage: 373.9 KB
    None 
    
    dataset (First 5 rows)
          AT      V       AP     RH      PE
    0  14.96  41.76  1024.07  73.17  463.26
    1  25.18  62.96  1020.04  59.08  444.37
    2   5.11  39.40  1012.16  92.14  488.56
    3  20.86  57.32  1010.24  76.64  446.48
    4  10.82  37.50  1009.23  96.62  473.90 
    
    Missing data processed.
    
    Categorical data encoded.
    
    Shape of x: (9568, 4)
    Shape of y: (9568, 1)
    Linear R2: -0.000
    Polynomial Regression R2: -0.004
    Decisiontree R2: -1.106
    Randomforest R2: -0.124
    Support Vector Regression R2: -0.052
    




    (-0.00040356999342305855,
     -0.00441916122154784,
     -1.105639211544076,
     -0.12388897783167252,
     -0.052240748469885245)



[Machine Learning A-Z Hands On Python and R in Data Science]: https://www.udemy.com/share/101Wci/
[here]: https://github.com/bleuong/bleuong.github.io/tree/master/assets/Jupyter_Notebooks/2020-10-24-Regression