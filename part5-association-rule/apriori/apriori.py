import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# train
from apyori import apriori
## 0.003 = 3 times * 7 weeks / 7501
rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length = 2, max_length=2)

# visualize
results = list(rules)

## putting results into pandas data frame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
# print(resultsinDataFrame)

# Displaying the results sorted by descending lifts
resultsinDataFrame = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
print(resultsinDataFrame)