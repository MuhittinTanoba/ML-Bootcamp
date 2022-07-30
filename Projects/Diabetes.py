import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("diabetes.csv")
df.head()
df.info()
df.isnull().sum()
for col in df.columns:
    print(col, df[col].nunique(), df[col].dtype)

# Grabbing column names
def grab_col_names(dataframe, car_th=20, cat_th=10):

    cat_cols = [col for col in dataframe.columns if (dataframe[col].dtype == "O") | (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() >car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int64", "float64"]) & (col not in cat_cols)]

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


##############################
## Analysis of Categorical Variables
##############################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


##############################
## Analysis of Numerical Variables
##############################

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist(bins=20)
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


# Mean of numerical variables relative to the target variable
for col in num_cols:
    print(df.groupby("Outcome").agg({col: ["mean"]}))

##############################
## Outliers
##############################

# Thresholding
def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3-quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))
# Only Insulin contains outliers


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_threshold(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


# ins_outliers_indexes = grab_outliers(df, "Insulin", index=True)
# for index in ins_outliers_indexes:
#     df.drop(index, axis=0, inplace=True)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    df_withouth_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_withouth_outliers

for col in num_cols:
    df = remove_outlier(df,col)

####################
# Missing Values
####################

df.isnull().values.any()
for col in num_cols:
    print(df.loc[df[col]==0].head(3))


def turn_0_value_into_nan(dataframe):
    zero_columns = [col for col in dataframe.columns if dataframe[col].nunique() > 20]

    for col in zero_columns:
        dataframe[col] = dataframe[col].replace(0, np.nan)


turn_0_value_into_nan(df)
df.isnull().sum()
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)

##############################
## Correlation
##############################

df.corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##############################
## Correlation
##############################

df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

df["NEW_BMI"] = pd.cut(x=df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healty", "Overweight", "Obese"])
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300],
                           labels=["Normal", "Prediabetes", "Diabetes"])

df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["Age"] >= 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_AGE_BMI_NOM"] = "healtymature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 18.5) & (df["BMI"] < 25), "NEW_AGE_BMI_NOM"] = "healtysenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 25) & (df["BMI"] < 30), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 25) & (df["BMI"] < 30), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 30), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 30), "NEW_AGE_BMI_NOM"] = "obesesenior"

df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]


##################################
# ENCODING
##################################


## Label Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

## One-Hot Encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()


##################################
# Normalization
##################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

##################################
# Modeling
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")


# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75


##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)

