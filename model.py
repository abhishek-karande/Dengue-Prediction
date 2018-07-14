import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
import xgboost
import math
#data loading
features=pd.read_csv("dengue_features_train.csv")
features=features.fillna(features.mean())
labels=pd.read_csv("dengue_labels_train.csv")
