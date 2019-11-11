import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve, auc, confusion_matrix

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

def plot_roc(y_true, y_proba):
    """
    y_true: true values
    y_proba: predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="curva ROC (área = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo")
    plt.ylabel("Taxa de Verdadeiro Positivo")
    plt.title("Curva ROC (Receiver operating characteristic)")
    plt.legend(loc="lower right")
    plt.show()
    pass
