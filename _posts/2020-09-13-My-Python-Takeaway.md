---
title:  "My Python Takeaway"
date:   2020-09-13 23:45:00 +0800
categories: Python
tags: libraries power_set generator exception loop exception object_oriented
---

# Python Recollection

I have started learning Python since 2019 and have amassed my notes in various forms such as text files, word documents, and Python files. Since I do not have have much opportunity to use Python in my job, I need to refer to these notes occassionally to refresh my memory.

I will take this opportunity to consolidate my notes here for ease of my future reference. The key focus will be on general Python. For specific libraries, I will create dedicated posts for them as I get experience in them. I will periodically update this post with new notes where applicable. So let's go.

## Libraries

1. [Scikit-Learn] (sklearn) - a key library for machine learning
2. [Matplotlib] (matplotlib) - visualisation library for Python
3. [Numpy] (numpy) - fundamental package for scientific computing in Python
4. [Pandas] (pandas) - Python analysis library
5. [Sympy] (sympy) - Python library for symbolic mathematics
6. [Statsmodel] (statsmodel) - Python module that provides estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration
7. [Random] (random) - Python standard library for random functions

## If-Elif-Else


```python
# An example of multiple assignment in Python within a single line of code
a, b = 10, 11

# Normal syntax
if a == b:
    print("a is equal to b")
elif a < b:
    print("a is smaller than b")
else:
    # While this is not a must to have, I prefer to always define else.
    # This ensure all cases are accounted and helps to identify errors.
    # If there is no code for it, I will just put "pass"
    print("a is bigger than b")

# An alternate syntax for simple if-else
# code_true if a==b else code_false_optional
print("a is bigger than b") if a > b else print("a is smaller than or equal to b")
```

    a is smaller than b
    a is smaller than or equal to b
    

## While Loop


```python
def demo_while(a, b):
    """
    This module demonstrate the while loop construct
    """

    print(f"For a = {a}, and b = {b}")

    while a < b:
        a += 2
        if a % 2 == 0:            
            print(f"{a} is divisible by 2")
            a += 1
        elif a % 3 == 0:
            print(f"{a} is divisible by 3")
            a += 2
            continue # jump back to the start of the loop to evaluate the condition
        else:            
            print(f"1 will be subtracted from {a}")
            a -= 1
    
        # break out of the loop if a becomes negative
        if a < 0:
            print("break out of the loop")
            break
        else:
            pass

demo_while(0, 10)
demo_while(5, 10)
demo_while(-10, 0)
```

    For a = 0, and b = 10
    2 is divisible by 2
    1 will be subtracted from 5
    6 is divisible by 2
    9 is divisible by 3
    For a = 5, and b = 10
    1 will be subtracted from 7
    8 is divisible by 2
    1 will be subtracted from 11
    For a = -10, and b = 0
    -8 is divisible by 2
    break out of the loop
    

## For Loop


```python
def demo_for(b):
    """
    This module demostrate the for loop construct
    """

    c = []
    
    for a in range(b):
        if a%2 == 1:
            continue
        elif a%5 == 4:
            break
        else:
            c.append(a)
    print(f"c = {c}")
    
    # Alternate for loop syntax combined with if-else loop
    d = [a for a in range(b) if a%3==0]
    
    print(f"d = {d}")
    
demo_for(10)
```

    c = [0, 2]
    d = [0, 3, 6, 9]
    

## Exception Handling


```python
from random import randint
# randint(a,b) would select a random integer from the range a to b inclusive

def exception_handling(a):
    """
    This module demonstrate how to raise different types of error
    """

    if a == 1:
        raise IndexError("This is an index error")
    elif a == 2:
        raise ValueError("This is a value error")
    elif a == 3:
        raise TypeError("This is wrong type")
    else:
        raise NameError("This is a name error")

exception_handling(randint(0, 5))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-24-489b076d3131> in <module>
         15         raise NameError("This is a name error")
         16 
    ---> 17 exception_handling(randint(0, 5))
    

    <ipython-input-24-489b076d3131> in exception_handling(a)
         11         raise ValueError("This is a value error")
         12     elif a == 3:
    ---> 13         raise TypeError("This is wrong type")
         14     else:
         15         raise NameError("This is a name error")
    

    TypeError: This is wrong type


## Dir method

Dir method returns a list of valid attributes of the object. Objects can be different data types or even functions and methods. It is a very useful method to identify the correct methods which can be used for the object.


```python
# a is a list defined as follow.
a = [1, "a"]

# get all attributes of a list
dir(a)
```




    ['__abs__',
     '__add__',
     '__and__',
     '__bool__',
     '__ceil__',
     '__class__',
     '__delattr__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__float__',
     '__floor__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__index__',
     '__init__',
     '__init_subclass__',
     '__int__',
     '__invert__',
     '__le__',
     '__lshift__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rlshift__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__round__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__trunc__',
     '__xor__',
     'as_integer_ratio',
     'bit_length',
     'conjugate',
     'denominator',
     'from_bytes',
     'imag',
     'numerator',
     'real',
     'to_bytes']




```python
# get a listing of Python built in functions
dir(__builtins__)
```




    ['ArithmeticError',
     'AssertionError',
     'AttributeError',
     'BaseException',
     'BlockingIOError',
     'BrokenPipeError',
     'BufferError',
     'BytesWarning',
     'ChildProcessError',
     'ConnectionAbortedError',
     'ConnectionError',
     'ConnectionRefusedError',
     'ConnectionResetError',
     'DeprecationWarning',
     'EOFError',
     'Ellipsis',
     'EnvironmentError',
     'Exception',
     'False',
     'FileExistsError',
     'FileNotFoundError',
     'FloatingPointError',
     'FutureWarning',
     'GeneratorExit',
     'IOError',
     'ImportError',
     'ImportWarning',
     'IndentationError',
     'IndexError',
     'InterruptedError',
     'IsADirectoryError',
     'KeyError',
     'KeyboardInterrupt',
     'LookupError',
     'MemoryError',
     'ModuleNotFoundError',
     'NameError',
     'None',
     'NotADirectoryError',
     'NotImplemented',
     'NotImplementedError',
     'OSError',
     'OverflowError',
     'PendingDeprecationWarning',
     'PermissionError',
     'ProcessLookupError',
     'RecursionError',
     'ReferenceError',
     'ResourceWarning',
     'RuntimeError',
     'RuntimeWarning',
     'StopAsyncIteration',
     'StopIteration',
     'SyntaxError',
     'SyntaxWarning',
     'SystemError',
     'SystemExit',
     'TabError',
     'TimeoutError',
     'True',
     'TypeError',
     'UnboundLocalError',
     'UnicodeDecodeError',
     'UnicodeEncodeError',
     'UnicodeError',
     'UnicodeTranslateError',
     'UnicodeWarning',
     'UserWarning',
     'ValueError',
     'Warning',
     'WindowsError',
     'ZeroDivisionError',
     '__IPYTHON__',
     '__build_class__',
     '__debug__',
     '__doc__',
     '__import__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'abs',
     'all',
     'any',
     'ascii',
     'bin',
     'bool',
     'breakpoint',
     'bytearray',
     'bytes',
     'callable',
     'chr',
     'classmethod',
     'compile',
     'complex',
     'copyright',
     'credits',
     'delattr',
     'dict',
     'dir',
     'display',
     'divmod',
     'enumerate',
     'eval',
     'exec',
     'filter',
     'float',
     'format',
     'frozenset',
     'get_ipython',
     'getattr',
     'globals',
     'hasattr',
     'hash',
     'help',
     'hex',
     'id',
     'input',
     'int',
     'isinstance',
     'issubclass',
     'iter',
     'len',
     'license',
     'list',
     'locals',
     'map',
     'max',
     'memoryview',
     'min',
     'next',
     'object',
     'oct',
     'open',
     'ord',
     'pow',
     'print',
     'property',
     'range',
     'repr',
     'reversed',
     'round',
     'set',
     'setattr',
     'slice',
     'sorted',
     'staticmethod',
     'str',
     'sum',
     'super',
     'tuple',
     'type',
     'vars',
     'zip']



## Defining New Class or Data Type

In object oriented programming, it is required to define your own class or data type. The following is an example of defining a parent class and a child class which inherits the parent class' properties. The Person class is the parent and the Male class is a child class which inherits the attributes of the parent class.


```python
class Person:
    """
    Person is the name of the class or data type
    Take note there is no bracket at the end of the class name
    Hence this is a parent class.
    """

    def __init__(self, name, age):
        """
        To initialise the object with the parameters which was passed in.

        Parameters
        ----------
        name : string
        age : integer

        Returns
        -------
        None.
        """
        self.name = name
        self.age = age
    
    def __str__(self):
        """
        To define the representation for printing
        """
        
        return f"Name: {self.name}, Age: {self.age}"
    
    # To define get methods to get the internal representation of the instance
    # Need as many get methods as there are variables to the class
    # In this case, need 2 method to get the name and age
    def getName(self):
        """
        To get the name of the class instance
        """
        
        return self.name
    
    def getAge(self):
        """
        To get the age of the class instance
        """
        
        return self.age

class Male(Person):
    """
    This is a child class from Person.
    The bracket is used to denote the parent class which the child would
    inherit the parent functions. Hence the child class do not need to
    repeat the functions.
    If the child class repeats the inherited functions, the child functions
    would override the parent function.
    """
    
    def __init__(self, name, age, gender="m"):
        """
        To initialise the gender
        
        Parameters
        ----------
        name : string
        age : integer
        gender : string (m/f)
        
        Returns
        -------
        None
        """
        
        self.gender = gender
        
        # Retain the rest of the initialisation from the Parent class
        # Have to pass the correct info accordingly
        super().__init__(name, age)
    
    def __str__(self):
        """
        To overwrite the Person representation for printing
        """
        
        return f"Name: {self.name}, Age: {self.age}, Gender: {self.gender}"
    
    def getGender(self):
        """
        To get the gender of class instance
        """
        
        return self.gender

    
# Create an object/instance of the Person class
p = Person("Sally", 21)
print(p)

# Create an instance of the Male Class
m = Male("John", 23)
# Use the representation defined by the Parent class
print(m)

# Use the get methods of the parent class
print(m.getName())
print(m.getAge())

# Use the get method of the child class
print(m.getGender())
```

    Name: Sally, Age: 21
    Name: John, Age: 23, Gender: m
    John
    23
    m
    

## Generator Methods

These are normal methods but with yield as the keyword instead of return. The method will stop at the next yield statement and return a value. It will continue with the next yield statement or until the end of the method. The key purpose is to return values progressively from the method instead of a big list at one go which require more memory. An example of a generator method is range. The following is an example of different generator methods to generate a power set of all possible combinations from the item list.


```python
def powerSet(items):
    """
    This version utilises a toolkit "itertools" inbuilt functions to generate
    the power set. The combinations method is able to generate different
    subsets (tuples) of s depending on desired length r.
    The chain.from_iterable() is used to link the various subsets together.
    The last for loop is used to convert the tuples into list.

    Inputs
    ------
    items : A list containing the items to be used for the combination

    Parameters
    ----------
    None.

    Outputs
    -------
    ele : A list containing the elements of the combination
    """
    import itertools as itt

    s = list(items)
    for ele in itt.chain.from_iterable(itt.combinations(s, r)
                                       for r in range(len(s)+1)):
        yield list(ele)


def powerSet1(items):
    """
    This version uses binary representation of the case number to select items.
    At the same index of items and binary representation, 1 would take and 0
    would not. This particular implementation uses binary bit manipulation to
    achieve the result.
    
    Inputs
    ------
    items : A list containing the items to be used for the combination

    Parameters
    ----------
    None.

    Outputs
    -------
    ele : A list containing the elements of the combination
    """
    N = len(items)
    # enumerate the 2**N possible combinations
    for i in range(2**N):
        ele = []
        for j in range(N):
            # test bit jth of integer i
            if (i >> j) % 2 == 1:
                ele.append(items[j])
        yield ele
        
        
def powerSet2(items):
    """
    Convert the case number into binary representation.
    Use the binary representation to determine which items will go into
    the set.
    In this version, the binary representation is given using string methods.
    
    Inputs
    ------
    items : A list containing the items to be used for the combination

    Parameters
    ----------
    N: number of items in the input list
    a_binary: binary representation of the combination serial number

    Outputs
    -------
    ele : A list containing the elements of the combination
    """

    s = list(items)
    N = len(items)
    for a in range(2**N):
        # convert case number into binary representation
        a_binary = list(("{0:0>"+str(N)+"b}").format(a))
        ele = []

        # use binary representation to select items in the set
        for i in range(N):
            if a_binary[i] == '1':
                ele.append(s[i])
        yield ele
        
        
def powerSet3(items):
    """
    This version uses recursive algorithm to generate the power sets.
    This is based on the search tree algorithm.
    
    Inputs
    ------
    items : A list containing the items to be used for the combination

    Parameters
    ----------
    None

    Outputs
    -------
    ele : A list containing the elements of the combination
    """

    # Fundamental case where there is only one item in the input list
    if len(items) == 1:
        yield items
        yield []
    else:
        for ele in powerSet3(items[1:]):
            yield [items[0]] + ele
            yield ele
            
            
# Test case
items = list(range(2))
print("Power Set")
print(list(powerSet(items)))

items = list(range(3))
print("Power Set 1")
print(list(powerSet1(items)))

items = list(range(4))
print("Power Set 2")
print(list(powerSet2(items)))

items = list(range(5))
print("Power Set 3")
print(list(powerSet3(items)))
```

    Power Set
    [[], [0], [1], [0, 1]]
    Power Set 1
    [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
    Power Set 2
    [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3], [0], [0, 3], [0, 2], [0, 2, 3], [0, 1], [0, 1, 3], [0, 1, 2], [0, 1, 2, 3]]
    Power Set 3
    [[0, 1, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [2, 3, 4], [0, 1, 3, 4], [1, 3, 4], [0, 3, 4], [3, 4], [0, 1, 2, 4], [1, 2, 4], [0, 2, 4], [2, 4], [0, 1, 4], [1, 4], [0, 4], [4], [0, 1, 2, 3], [1, 2, 3], [0, 2, 3], [2, 3], [0, 1, 3], [1, 3], [0, 3], [3], [0, 1, 2], [1, 2], [0, 2], [2], [0, 1], [1], [0], []]
    


```python

```

[Scikit-Learn]: https://scikit-learn.org/stable/
[Matplotlib]: https://matplotlib.org/
[Numpy]: https://numpy.org/
[Pandas]: https://pandas.pydata.org/
[Sympy]: https://www.sympy.org/en/index.html
[Statsmodel]: https://www.statsmodels.org/stable/index.html
[Random]: https://docs.python.org/3/library/random.html
