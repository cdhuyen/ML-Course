import numpy as np
import pandas as pd
from scipy.io import loadmat


mat = loadmat("./data/lung.mat")
# Merge to dataframe
data = pd.DataFrame(np.concatenate((mat["X"], mat["Y"]), axis=1))

# Delete instances with class label 5
filter = data.iloc[:, -1] != 5
data = data[filter]

data.to_csv("./data/lung.csv", header=False, index=False)

