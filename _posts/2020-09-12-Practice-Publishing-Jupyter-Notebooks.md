---
title:  "Practice Publishing Jupyter Notebooks"
date:   2020-09-12 14:53:57 +0800
categories: General
tags: Jupyter NBconvert
---

Since my aim was to document my learning journey, the easiest would be to use Jupyter notebooks and publish them on my blog.

Hence I have created a Jupyter notebook using the Jupyter module in Anaconda. I convert a Data Preprocessing module which I had written in Spyder as a sample Jupyter notebook.

Then I used the [NBconvert] function to render my notebook into markdown format as shown below.

The main command is:

{% highlight ruby %}
jupyter nbconvert --to markdown NOTEBOOK-NAME.ipynb
{% endhighlight %}

Here is my first Jupyter notebook. The actual notebook and sample file can be accessed [here](https://github.com/bleuong/bleuong.github.io/tree/master/assets/Jupyter_Notebooks/2020-09-12-Data-Preprocessing/).

# Data Preprocessing

I created a reuseable module to preprocess the data file as the first step in any data analysis. It comprise of the following:

1. Import data from file
2. Take care of missing data
3. Encode categorical data if required

## Import required libraries


```python
import numpy as np  # library for numerical manimuplation
import matplotlib.pyplot as plt  # library for plotting
import pandas as pd  # library for importing datasets
```

## Actual Data Preprocessing Module


```python
def Data_Preprocessing(InputFileName=None,
                       ImputerInput={"selectedCol": None, "strategy": "mean"},
                       xEncoderInput={"cols": None, "AvoidDummy": False},
                       yEncoderInput=False):
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
    InputFileName : This is the name of the file to be imported for analysis.
                        The file must be located in the same folder as this
                        python script.
    ImputerInput : A dict containing a list of columns to fill in the missing
                    values and the strategy for the missing values.
                    Can be "mean", "median", "most_frequent". If the list
                    of columns is None, it means no missing values.
    xEncoderInput : A dict containing a list of columns index in x to encode
                        the categorical data and a boolen value to indicate if
                        there is a need to avoid the dummy variable trap. An
                        empty list indicate no need to perform encoding.
                        Default is {cols: [], AvoidDummy=False}
    yEncoderInput : A boolen value indicating if the dependent variable y is
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
    from os import listdir, getcwd
    from os.path import isfile, join
    from sys import exit

    # Get file name if it is not provided
    if InputFileName is None:
        while True:
            print(f"Current working directory is:\n{getcwd()}")
            file_list = [f for f in listdir() if isfile(join(f))]
            file_list_index = list(range(len(file_list)))
            print("Files available are:")
            for fli in file_list_index:
                print(f"({fli}) {file_list[fli]}")
            ui = input("Please select the corresponding file or -1 to exit ",
                       "if file is not found.\nPlease ensure data file is in ",
                       "the same directory as the code file.")
            try:
                # Check if ui is correctly input
                ui = int(ui)
                # Input is an integer. Check if it falls in the correct range.
                if ui == -1:
                    # Exit
                    exit
                elif ui in file_list_index:
                    # Valid option selected. Verify selection
                    ui1 = input(f"You have selected {file_list[ui]}.\n",
                                "Please confirm Y/N.")
                    if ui1.upper() == "Y":
                        # Correct selection. break out of while loop
                        InputFileName = file_list[ui]
                        break
                    elif ui1.upper() == "N":
                        # Incorrect selection
                        pass
                    else:
                        # Unknown input
                        print("You have provided an invalid selection.")
                else:
                    # Incorrect input. Will raise error exception
                    print("You have provided an invalid selection.")
            except ValueError:
                # Wrong user input
                print("You have provided an invalid selection.")
    else:
        # Date file name provided. Do nothing
        pass

    # Read data from data file
    dataset = pd.read_csv(InputFileName)
    print("Data imported successfully.\n")

    print(dataset.info(), "\n")
    print(dataset.head())

    # Extract the dependent variable (y) and independent features (x).
    # It is assumed the dependent variable column is the last one.
    # All rows are selected. All columns except last one is selected.
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    """ Take care of missing data """
    # Check if there is missing data to be filled in
    if ImputerInput["selectedCol"] is not None:
        # User has provided columns index which has missing data
        from sklearn.impute import SimpleImputer

        # Create the imputer object to be used on x, y
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy=ImputerInput["strategy"])

        # Cycle through every column index provided by user
        for colIndex in ImputerInput["selectedCol"]:
            # Link the imputer object to the numerical columns in x with the
            # correct column index
            # Need to reshape (-1, 1) each column for the imputer object to
            # work and reshape it back (1, -1) to put back into x.
            imputer.fit(x[:, colIndex].reshape(-1, 1))
            # Perform the actual replacement on x
            x[:, colIndex] = \
                imputer.transform(x[:, colIndex].reshape(-1, 1)).reshape(1, -1)
    else:
        # The selectedCol list is None. Hence no missing data.
        pass

    """ Encode categorical data """
    # Check if there is categorical data in x to encode
    if xEncoderInput["cols"] is not None:
        # Column index had been provided by user to encode
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        # Cycle through each selected column index for encoding
        for selectedCol in xEncoderInput["cols"]:
            # Create ColumnTransformer object
            ct = ColumnTransformer(transformers=[("encoder",
                                                  OneHotEncoder(),
                                                  [selectedCol])],
                                   remainder="passthrough")
            # Fit and transform the ct object to x in one step
            # Need to use np.array to force fit the result from fit_transform
            # into a numpy array for subsequent analysis
            x = np.array(ct.fit_transform(x))

            # Remove first column of dummy variable to avoid dummy variable
            # trap. The dummy variables are arranged in alphabetical order.
            if xEncoderInput["AvoidDummy"]:
                # Need to avoid dummy variable trap because the model is unable
                # to take care of the dummy variable
                x = x[:, 1:]
            else:
                # Model is able to avoid dummy variable trap. Hence no need to
                # remove any dummy variable.
                pass
    else:
        # User did not provide any columns for encoding. So do nothing.
        pass

    # Check if need to encode dependent variable y
    if yEncoderInput:
        # Dependent variable y is categorical and need to be encoded
        from sklearn.preprocessing import LabelEncoder

        # Create LabelEncoder object
        le = LabelEncoder()
        # Fit_transfer LabelEncoder onto dependent variable y
        y = le.fit_transform(y)
    else:
        # Dependent variable y is not categorical.
        pass

    return x, y
```

## Main Module to Utilise the Data Pre-Processing Module


```python
def Part01():
    """ Data Preprocessing Inputs """
    # Data file name. File must be in same folder as script
    Dataset = "Part01Section02Data.csv"
    # Missing numerical data to be filled up
    # Default values "selectedCol": [],"strategy": "mean"
    ImputerInput = {"selectedCol": [1, 2], "strategy": "mean"}
    # Columns in x with categorical data to encode, default {"cols": [],
    #                                                   "AvoidDummy": False}
    xEncoderInput = {"cols": [0], "AvoidDummy": False}
    # Indicate if need to encode dependent variable y, enter True or False
    yEncoderInput = True

    x, y = Data_Preprocessing(Dataset, ImputerInput, xEncoderInput,
                              yEncoderInput)

    return x, y
```

## Calling the module


```python
x, y = Part01()
print(x)
```

    Data imported successfully.
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Country    10 non-null     object 
     1   Age        9 non-null      float64
     2   Salary     9 non-null      float64
     3   Purchased  10 non-null     object 
    dtypes: float64(2), object(2)
    memory usage: 448.0+ bytes
    None 
    
       Country   Age   Salary Purchased
    0   France  44.0  72000.0        No
    1    Spain  27.0  48000.0       Yes
    2  Germany  30.0  54000.0        No
    3    Spain  38.0  61000.0        No
    4  Germany  40.0      NaN       Yes
    [[1.0 0.0 0.0 44.0 72000.0]
     [0.0 0.0 1.0 27.0 48000.0]
     [0.0 1.0 0.0 30.0 54000.0]
     [0.0 0.0 1.0 38.0 61000.0]
     [0.0 1.0 0.0 40.0 63777.77777777778]
     [1.0 0.0 0.0 35.0 58000.0]
     [0.0 0.0 1.0 38.77777777777778 52000.0]
     [1.0 0.0 0.0 48.0 79000.0]
     [0.0 1.0 0.0 50.0 83000.0]
     [1.0 0.0 0.0 37.0 67000.0]]
    


```python
print(y)
```

    [0 1 0 0 1 1 0 1 0 1]
    


```python

```

[NBconvert]: https://docs.github.com/en/github/managing-files-in-a-repository/working-with-jupyter-notebook-files-on-github
