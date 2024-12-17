# -*- coding: utf-8 -*-

# importing libs# importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno  ModuleNotFoundError: No module named 'numpy.rec'
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import folium
from folium.plugins import HeatMap
import plotly.express as px

# plotting configurations
plt.style.use('fivethirtyeight')
# %matplotlib inline
pd.set_option('display.max_columns', 32)


# reading data
df = pd.read_csv('../data/hv.csv')
df = df.iloc[10:]
print(df.head(10))


df = df["DROP TABLE"].str.split('|', expand=True)
print(df.head(10))

df.rename(columns={0: 'time', 1: 'device_name', 2: 'key', 3: 'value'}, inplace=True)


df_sample = df.head(10000)

unique_values = df_sample['key'].unique()























