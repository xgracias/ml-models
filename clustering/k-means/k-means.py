import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values