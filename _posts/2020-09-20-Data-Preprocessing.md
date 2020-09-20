---
title:  "Data Preprocessing"
date:   2020-09-20 21:17:00 +0800
categories: Preprocessing
tags: module pandas categorical encode missing imputer feature_scaling split_data
---

# Codifying My Knowledge

In one of my previous posts, I had put up a data pre-processing module which I had developed as I commenced my study in Data Analytics. It was merely used as a test. Today, I shall clearly put down my rationale for the way I had written my module.

First and foremost, in all the codes which I would be sharing, they are written in modular format. This means they are written as functions instead of single executable lines of code. The main reason for doing so is to allow me to reuse the code. My aim was to develop individual functions in a single file, which would allow me to utiise them in anyway in future in my data analytics work. With these modules, I can call upon any of them depending on what type of analysis I would do. That is how I am aiming to codify my knowledge and faciliate application in future work.


## Single line of code


```python
print("Hello World.")
```

    Hello World.
    

## Modular code


```python
def say_hi(name, age):
    """
    To say hi to the world.
    """

    print(f"Hello world. My name is {name} and I am {age} years old.")
```


```python
say_hi("John", 12)
```

    Hello world. My name is John and I am 12 years old.
    


```python
say_hi("Sally", 51)
```

    Hello world. My name is Sally and I am 51 years old.
    

In addition, I had configured my IDE (Spyder) to follow [PEP-8], which is the recommended code writing convention for Python. There are still some convention which is manually followed such as small letters separated by underscores for function names and variable names.

# Data Pre-Processing

Data pre-processing typically comprises of 5 main steps:
1. Read data into python readable format (typically is pandas)
2. Fill in missing data according to desired strategy (mean, median, most_frequent) (optional)
3. Encode categorical data in both x and y (optional)
4. Split data into training set and test set (optional)
5. Scale the data if the model is unable to handle extreme values (optional)

I have bundled the first 3 steps into a single module and the remaining steps as their respective modules. The main reason is because there is a change in the data format when we start to perform Steps 4 and 5. From Steps 1 to 3, I would be working with x, y. After Step 4, I would have x_train, y_train, x_test, y_test. After Step 5, I would additionally have scaled version of the training and test set and also the scaler object to inverse the scaling. By bundling the above steps into 3 different modules, I would be able to call on any of them depending on the model to be used.

### Data Pre-Processing Module


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
```

### Split Dataset Module


```python
def split_dataset(x, y, split_data_input=[0.2, None]):
    """
    This is to split the dataset into the training set and test set.

    Inputs
    ------
    x : numpy array of independent features
    y : a list of dependent variable
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

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=split_data_input[0],
                         random_state=split_data_input[1])

    # Check if x_train and x_test are 2 dimensional array. Reshape if necessary
    if len(x_train.shape) == 1:
        # 1 dimensional list. Reshape
        x_train = x_train.reshape(len(x_train), 1)
    else:
        # x_train is 2 dimensional array. Do nothing
        pass

    if len(x_test.shape) == 1:
        # 1 dimensional list. Reshape
        x_test = x_test.reshape(len(x_test), 1)
    else:
        # x_test is 2 dimensional array. Do nothing
        pass

    return x_train, y_train, x_test, y_test
```

### Feature Scaling Module


```python
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

### Main Control Module to Process the Data


```python
def main():
    # Data file name. File must be in same folder as script
    dataset = "sample.csv"
    # Missing numerical data to be filled up
    # Default values "selected_col": [],"strategy": "mean"
    imputer_input = {"selected_col": [], "strategy": "mean"}
    # Columns in x with categorical data to encode, default {"cols": [],
    #                                                   "AvoidDummy": False}
    x_encoder_input = {"cols": [], "AvoidDummy": False}
    # Indicate if need to encode dependent variable y, enter True or False
    y_encoder_input = False

    x, y = data_preprocessing(dataset, imputer_input, x_encoder_input,
                              y_encoder_input)

    # Inputs for splitting the dataset
    # State the random_state as 1 for troubleshooting. Use None for random.
    split_data_input = [0.2, 0]

    # Always split dataset first prior to Feature scaling because you do not
    # want the mean value for the test set to be influenced by the training set
    x_train, y_train, x_test, y_test = split_dataset(x, y, split_data_input)

    # Set up the columns which needed to be scaled
    # scaler_input = []
    # # Execute the scaling
    # x_train_sc, y_train_sc, sc_x_train, sc_y_train, x_test_sc, y_test_sc = \
    #     Features_Scaling(x_train, None, scaler_input, x_test, None)

    print("x_train\n", x_train)
    print("y_train\n", y_train)
    print("x_test\n", x_test)
    print("y_test\n", y_test)
    
    # To continue with model building (to be covered in subsequent posts)
```


```python
main()
```

    Data imported successfully.
    
    dataset Info
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30 entries, 0 to 29
    Data columns (total 2 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   YearsExperience  30 non-null     float64
     1   Salary           30 non-null     float64
    dtypes: float64(2)
    memory usage: 608.0 bytes
    None 
    
    dataset (First 5 rows)
       YearsExperience   Salary
    0              1.1  39343.0
    1              1.3  46205.0
    2              1.5  37731.0
    3              2.0  43525.0
    4              2.2  39891.0 
    
    Missing data processed.
    
    Categorical data encoded.
    
    x_train
     [[ 9.6]
     [ 4. ]
     [ 5.3]
     [ 7.9]
     [ 2.9]
     [ 5.1]
     [ 3.2]
     [ 4.5]
     [ 8.2]
     [ 6.8]
     [ 1.3]
     [10.5]
     [ 3. ]
     [ 2.2]
     [ 5.9]
     [ 6. ]
     [ 3.7]
     [ 3.2]
     [ 9. ]
     [ 2. ]
     [ 1.1]
     [ 7.1]
     [ 4.9]
     [ 4. ]]
    y_train
     [112635.  55794.  83088. 101302.  56642.  66029.  64445.  61111. 113812.
      91738.  46205. 121872.  60150.  39891.  81363.  93940.  57189.  54445.
     105582.  43525.  39343.  98273.  67938.  56957.]
    x_test
     [[ 1.5]
     [10.3]
     [ 4.1]
     [ 3.9]
     [ 9.5]
     [ 8.7]]
    y_test
     [ 37731. 122391.  57081.  63218. 116969. 109431.]
    

### Lesson Learnt

In developing these modules, the arguements to be passed between the modules must be scrutinised. Other than data type, the other concern is the shape of the pandas array used to store the data. Different models may have different requirements for the shape, whether it is a one column or one row array. Sometimes, the models can accommodate to it and would only give a warning. Otherwise, it may cause the program to malfunction. So, do consider to include the shape of the array as part of the docstring to remind oneself to take note.

You can find the Jupyter notebook for these modules [here].

[PEP-8]: https://www.python.org/dev/peps/pep-0008/
[here]: https://github.com/bleuong/bleuong.github.io/tree/master/assets/Jupyter_Notebooks/2020-09-20-Data-Preprocessing/