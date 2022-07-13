# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
# Görev 10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
# Görev 12: who değişkenini dataframe’den çıkarınız.
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
# setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz.
# Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.
# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz.
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit
# olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
# Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır. Parametre olarak cinsiyet ve total_bill
# alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)
# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.
# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic") #Gorev1
df["sex"].value_counts() #Gorev2

###### Gorev3 ######
for col in df.columns:
    print(f"Sutun:{col} Unique değer sayısı: {df[col].nunique()}")
####################

df["pclass"].nunique() #Gorev4
df[["pclass", "parch"]].nunique() #Gorev5

###### Gorev6 ######
df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtypes
####################

df.loc[(df["embarked"] == "C")] #Gorev7
df.loc[~(df["embarked"] == "S")] #Gorev8
df.loc[(df["sex"] == "female") & (df["age"] < 30)].head() #Gorev9
df.loc[(df["fare"] > 500) | (df["age"] > 70)].head() #Gorev10

df.isnull().sum() #Gorev11
df.drop("who", axis=1) #Gorev12
df["deck"].fillna(df["deck"].mode()[0]) #Gorev13
df["age"].fillna(df["age"].median()) #Gorev14

###### Gorev15 ######
df.groupby("survived").agg({"pclass":["sum", "count", "mean"]})
df.groupby("sex").agg({"survived":["sum", "count", "mean"]})
#####################

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0) #Gorev16

df = sns.load_dataset("tips") #Gorev17
df.groupby("time").agg({"total_bill": ["sum", "min", "max"]}) #Gorev18
df.groupby(["time", "day"]).agg({"total_bill": ["sum", "min", "max"]}) #Gorev19

###### Gorev20 ######
new_df = df.loc[(df["sex"] == "Female") & (df["time"] == "Lunch")]
new_df.groupby("day").agg({"total_bill": ["sum", "min", "max"],
                           "tip": ["sum", "min", "max"]})
#####################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean() #Gorev21
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"] #Gorev22

###### Gorev23 ######
mean_male = df.loc[(df["sex"] == "Male")]["total_bill"].mean()
mean_female = df.loc[(df["sex"] == "Female")]["total_bill"].mean()

def total_bill_flag(gender, total_bill):
    if gender == "Female" and total_bill < mean_female:
        return 0
    elif gender == "Female" and total_bill >= mean_female:
        return 1
    elif gender == "Male" and total_bill < mean_male:
        return 0
    elif gender == "Male" and total_bill >= mean_male:
        return 1

df["total_bill_flag"] = df.apply(lambda x: total_bill_flag(x.sex, x.total_bill), axis=1)
####################

df.groupby(["total_bill_flag","sex"]).total_bill_flag.value_counts() #Gorev24
new_df = df.sort_values(by=["total_bill_tip_sum"], ascending=False).head(30) #Gorev25