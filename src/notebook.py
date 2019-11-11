#!/usr/bin/env python
# coding: utf-8

# 1. ler bases:
# userprofile
# rating_final
# chefmozcuisine


import pandas as pd
users = pd.read_csv("data/userprofile.csv")
ratings = pd.read_csv("data/rating_final.csv")
cuisine = pd.read_csv("data/chefmozcuisine.csv")


# 2. selectionar tipo restaurante
# ver frequencia dos restaurantes: escolher mexicano
# pegar os ratings dos mexicanos

mexican_cuisine = "Mexican"
mexican_cuisine = cuisine.loc[cuisine["Rcuisine"] == mexican_cuisine, "placeID"].values.tolist()
mexican_ratings = ratings.loc[ratings["placeID"].isin(mexican_cuisine), ]


# 3. análise exploratória

import matplotlib.pyplot as plt
mexican_ratings["rating"].plot(kind="hist")
plt.show();
mexican_ratings["food_rating"].plot(kind="hist")
plt.show();
mexican_ratings["service_rating"].plot(kind="hist")
plt.show();

# 4. alvo

grouped_ratings = mexican_ratings.groupby("userID", as_index=False)["rating"].mean()
grouped_ratings.loc[:, "gosta_mexicano"] = 0
grouped_ratings.loc[grouped_ratings["rating"] == 2, "gosta_mexicano"] = 1
del grouped_ratings["rating"]

# 5. vetorizando a base de usuários

def vectorize(users):
    """
    Parameters
    ----------
    users: pd.DataFrame

    Return
    ------
    users: pd.DataFrame
        A vectorized version of the original pd.DataFrame
        Text columns are turned into number columns
    """
    variaveis_categoricas = users.select_dtypes(include="object").columns.tolist()
    variaveis_categoricas.remove("userID")
    users.loc[:, variaveis_categoricas] = users.loc[:, variaveis_categoricas].astype("category")
    for variavel in variaveis_categoricas:
        users.loc[:, variavel] = users[variavel].cat.codes
    return users


# 6. treino e teste

train = pd.merge(users, grouped_ratings, how="inner", on="userID")
usuarios_base_treinamento = train["userID"].values.tolist()
test = users.loc[~users["userID"].isin(usuarios_base_treinamento), ]


# 7. modelo

from sklearn.linear_model import LogisticRegression
X_columns = train.columns.tolist()
X_columns.remove("userID")
X_columns.remove("gosta_mexicano")
y_column = "gosta_mexicano"
X = train[X_columns]
y = train[y_column]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto", random_state=42)
clf.fit(X_train, y_train)
usuarios = test[["userID"]].copy()
usuarios.loc[:, "Gosta de mexicano?"] = clf.predict(X_test)
usuarios.loc[:, "Probabilidade de gostas de mexicano"] = clf.predict_proba(X_test)[:, 1]


# 8. performance

from sklearn.metrics import accuracy_score, average_precision_score, roc_curve, auc, confusion_matrix
accuracy_score(y_test, predictions)

def plot_roc(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, previsoes_probabilidade, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color="darkorange"", lw=lw, label=""curva ROC (área = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo")
    plt.ylabel("Taxa de Verdadeiro Positivo")
    plt.title("Curva ROC (Receiver operating characteristic)")
    plt.legend(loc="lower right")
    plt.show()
    pass

