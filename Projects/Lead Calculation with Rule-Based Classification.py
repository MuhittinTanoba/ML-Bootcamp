import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("persona.csv")
df.head()
df.info()
df.describe().T
df.columns

# GOREV 1
# Number of unique SOURCE?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Number of unique PRICE?
df["PRICE"].nunique()
df["PRICE"].value_counts()

# How many sales from which country?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY").agg({"PRICE":"sum"})

# What are the sales numbers by SOURCE types?
df["SOURCE"].value_counts()

# What are the PRICE averages by COUNTRY?
df.groupby("COUNTRY").agg({"PRICE":"mean"})

# What are the PRICE averages by SOURCE?
df.groupby("SOURCE").agg({"PRICE":"mean"})

# What are the PRICE averages by COUNTRY-SOURCE?
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE":"mean"})


# GOREV 2
# What are the PRICE averages by COUNTRY-SOURCE-SEX-AGE?
df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"]).agg({"PRICE":"mean"})


# GOREV 3
# Sorted by PRICE.
agg_df = df.sort_values("PRICE", ascending=False)

# GOREV 4
agg_df.reset_index(drop=True, inplace=True)

# GOREV 5
# Converted AGE variable to categorical variable.
bins = [0, 18, 23, 30, 40, 70]
group_names = ["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGE_CAT"] = pd.cut(agg_df.AGE, bins, labels = group_names)

# GOREV 6
# Persona defined. Duplicated data removed and price averaged
customer_level_based = [f"{col['COUNTRY'].upper()}_{col['SOURCE'].upper()}_{col['SEX'].upper()}_{col['AGE_CAT'].upper()}" for index, col in agg_df.iterrows()]
agg_df["customer_level_based"] = customer_level_based
average_by_price = agg_df.groupby('customer_level_based').agg({"PRICE": "mean"})
average_by_price.reset_index(inplace=True)
average_by_price.sort_values("PRICE", inplace=True, ignore_index=True, ascending=False)

# GOREV 7
# Personas segmented
average_by_price["SEGMENT"] = pd.qcut(average_by_price["PRICE"], 4, labels=["D", "C", "B", "A"])
average_by_price.groupby('SEGMENT').agg({"PRICE": ["sum", "mean", "max"]})

# GOREV 8
new_user = "TUR_ANDROID_FEMALE_31_40"
average_by_price[average_by_price["customer_level_based"] == new_user]
new_user2 = "FRA_IOS_FEMALE_31_40"
average_by_price[average_by_price["customer_level_based"] == new_user2]
