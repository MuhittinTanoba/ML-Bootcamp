import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Görev:  cat_summary() fonksiyonuna 1 özellik ekleyiniz.
# Bu özellik argümanla biçimlendirilebilir olsun.
# Var olan özelliğide argümanla kontrol edilebilir hale getirebilirsiniz.

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset('titanic')
df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
     """
    Veri setindeki kategorik, numerik ve kategorik fakat kordinal değişkenlerin isimlerini verir

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir
    cat_th: int, float
        numerik fakat kategorik olanların eşik değeridir
    car_th: int, float
        kategorik fakat kordinal değişkenler için sınıf eşik değeridir.

    Returns
    -------
    cat_cols: list
        kategorik değişkenlerin listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kordinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içindedir

    """

     cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

     num_but_cat = [col for col in dataframe.columns if
                    dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

     cat_but_car = [col for col in dataframe.columns if
                    dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

     cat_cols = cat_cols + num_but_cat
     cat_cols = [col for col in cat_cols if col not in cat_but_car]

     num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
     num_cols = [col for col in num_cols if col not in cat_cols]

     return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False, cat_desc=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

    if cat_desc:
        print(f"Number of Categorical Variables:",{len(cat_cols)})
        for col in cat_cols:
            print(col)

for col in cat_cols:
    cat_summary(df, col, cat_desc=True)
###########################################################################

#Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi (uygunsa) barındıran numpy tarzı docstring
#yazınız. (task, params, return, example)

def check_df(dataframe, head=5):
    """
    DataFrame hakkında bilgi verir
    Parameters
    ----------
    dataframe: dataframe
        Bilgi almak istediğimiz dataframe.
    head: int
        Baştan kaç tane değişken ile ilgili bilgi almak istiyorsak onu bu parametreye veririz.

    -------

    """
    print("----Shape---")
    print(dataframe.shape)
    print("----Types----")
    print(dataframe.dtypes)
    print("----Head----")
    print(dataframe.head(head))
    print("----Tail----")
    print(dataframe.tail(head))
    print("----NA----")
    print(dataframe.isnull().sum())
    print("----Quantiles----")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False, cat_desc=False):
    """
    Kategorik değişkenlerin sayılarını ve oranlarını verir.
    Parameters
    ----------
    dataframe: dataframe
        Verilerin bulunduğu dataframe.
    col_name: String
        Analiz etmek istediğimiz değişkenin ismi.
    plot: Boolean
        Grafik çizdirme.
    cat_desc: Boolean
        Kategorik değişkenlerin adedini ve isimlerini yazdırır.
    -------

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

    if cat_desc:
        print(f"Number of Categorical Variables:",{len(cat_cols)})
        for col in cat_cols:
            print(col)


