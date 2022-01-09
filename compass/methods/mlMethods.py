from skelm import ELMClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def myModel(mlMethod):
    """
    This is the model method that allows to select the desired ML classifier (for binary classification in this version)
    :param mlMethod: string
    :return: ML model

    """


    if mlMethod == "RLM":
        model = linear_model.LinearRegression()

    if mlMethod == "LDA":
        model = linear_model.LinearDiscriminantAnalysis(solver="eigen")

    if mlMethod == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    if mlMethod == "LR":
        model = linear_model.LogisticRegression(solver='lbfgs', random_state=7, max_iter=200)

    if mlMethod == "SVM":
        model = SVC(gamma='auto', probability=True, random_state=42)

    if mlMethod == "GNB":
        model = linear_model.GaussianNB()
    if mlMethod == "BR":
        model = linear_model.BayesianRidge()
    if mlMethod == "RID":
        model = linear_model.Ridge(alpha=.5)
    if mlMethod == "ELAS":
        model = linear_model.ElasticNet(alpha=.5)
    if mlMethod == "LASSO":
        model = linear_model.Lasso(alpha=.5)

    if mlMethod == "ELM":
        model = ELMClassifier(n_neurons=10, ufunc='sigm')

    return model
