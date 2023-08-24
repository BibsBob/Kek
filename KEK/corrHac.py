import numpy as np
from sklearn.linear_model import LinearRegression
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import Perceptron
import statsmodels.api as sm


def checkMultiCor():
    # importing the data
    df = pandas.read_csv("csvFile/new1.csv")

    print(df.corr())

    # plotting the correlation heatmap
    df_plot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

    # displaying the heatmap
    #column_to_remove = 'Row#'
    #df.drop(column_to_remove, inplace=True, axis=1)
    #df.to_csv('csvFile/new1.csv', index=False)
    plt.show()


def loadModel(modelName):
    dataset = pandas.read_csv('csvFile/new1.csv').astype(int)
    #print(dataset.head())
    # разделение данных на атрибуты и метки
    X = dataset[
        ['clonesize', 'bumbles', 'osmia', 'MaxOfUpperTRange',

         'AverageRainingDays', 'seeds']]
    y = dataset['yield']

    # Делим на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if modelName == 'multLineReg':
        multLineReg(X, X_train, X_test, y_train, y_test)
    else:
        if modelName == 'decisionTree':
            decisionTree(X_train, X_test, y_train, y_test)
        else:
            if modelName == 'perceptron':
                perceptron(X_train, X_test, y_train, y_test)
            else:
                if modelName == 'OSL':
                    OSL()
                else:
                    print('Not correct model name')
                    exit(102)


def OSL():
    dataset = pandas.read_csv('csvFile/new1.csv')
    X = dataset[
        ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange', 'MaxOfLowerTRange', 'MinOfLowerTRange',
         'RainingDays', 'AverageRainingDays', 'fruitmass', 'fruitset', 'Bear', 'WindSpeed', 'seeds']]

    y = dataset['yield']
    print(dataset['AverageRainingDays'])

    # Делим на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    x_incl_cons = sm.add_constant(X_train)
    model = sm.OLS(y_train, x_incl_cons)  # ordinary least square
    results = model.fit()  # regresssion results

    print(round(results.pvalues,3))


def multLineReg(X, X_train, X_test, y_train, y_test):
    # Создаем модель
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Смотрим коэфы для атрибутов

    # Делаем предикт и сравниваем с реальной оценкой (np.rint - округляет)
    y_pred = np.rint(regressor.predict(X_test))
    df = pandas.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df)
    df.to_csv('csvFile/out1.csv', index=False)


def decisionTree(X_train, X_test, y_train, y_test):
    dataset = pandas.read_csv('csvFile/new1.csv').astype(int)

    print(dataset.head())
    # разделение данных на атрибуты и метки
    X = dataset[
        ['clonesize', 'bumbles', 'osmia', 'MaxOfUpperTRange',
         'AverageRainingDays', 'seeds']]

    y = dataset['yield']

    # Делим на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Обучаем модель
    model = tree.DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)

    # Оценка модели
    print('Оценка: ', model.score(X_train, y_train))

    # Делаем предикт
    y_pred = model.predict(X_test)

    # Записываем предикт
    df = pandas.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df)
    df.to_csv('csvFile/out2.csv', index=False)


def perceptron(X_train, X_test, y_train, y_test):
    model = Perceptron(tol=1e-3, random_state=0)
    model.fit(X_train, y_train)

    print('Оценка: ', model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    df = pandas.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df)
    df.to_csv('csvFile/out3.csv', index=False)


if __name__ == '__main__':
    checkMultiCor()
    # multLineReg decisionTree perceptron
    loadModel('multLineReg')    # Множественная линейная регрессия
    loadModel('decisionTree')   # Модель - дерево решений
    loadModel('perceptron')     # Модель - персептрон
    loadModel('OSL')            # Подсчет p-value