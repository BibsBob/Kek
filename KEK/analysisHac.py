import pandas as pd
import matplotlib.pyplot as plt

def analysis():
    cleaning('csvFile/out1.csv')    # Анализ Множественной линейной регрессии
    cleaning('csvFile/out2.csv')    # Анализ Дерева решений
    cleaning('csvFile/out3.csv')    # Анализ Персептрона

    chart()                         # График Предсказанных значений и реальных


def cleaning(csvname):
    dataset = pd.read_csv(csvname)
    df = pd.DataFrame(dataset)

    df = df[df['Predicted'] >= 0]

    # Считаем отклонение
    df['difference'] = abs(df['Actual'] - df['Predicted'])

    df.to_csv(csvname, index=False)

    truepred = 0
    # Считаем метрики
    for yeild in df['difference']:
        if yeild <= 350:
            truepred = truepred + 1    # Количество точных предсказаний
    proc = truepred / 156 * 100
    maxdif = df['difference'].max()             # Максимальное отклонение
    average = df['difference'].mean()           # Среднее значение погрешности

    mape = (1/df.shape[0])*sum(abs(df['Actual']-df['Predicted'])/df['Actual'])*100

    print(csvname)
    print('___________________________________________________________________________________________________')
    print('Количество точных предсказаний: ', truepred, 'Процент:', truepred/df.shape[0])
    print('Максимальное отклонение: ', maxdif)
    print('Среднее значение отклонения: ', average)
    print('MAPE (средня абсолютная ошибка в процентах): ', mape)
    print('___________________________________________________________________________________________________')
    print('')


def chart():
    # Строим график предиктов
    # dataset = pd.concat(map(pd.read_csv, ['csvFile/out1.csv', 'csvFile/out2.csv', 'csvFile/out3.csv']), ignore_index=True)

    dataset = pd.read_csv('csvFile/out1.csv')   # Множественная ЛГ
    #dataset = pd.read_csv('csvFile/out2.csv')  # Дерево
    #dataset = pd.read_csv('csvFile/out3.csv')  # Персептрон
    df = pd.DataFrame(dataset)

    # Объединить в один график
    #df['Predicted2'] = dataset2['Predicted']
    #df['Predicted3'] = dataset3['Predicted']
    #df.plot(y=["Predicted", "Predicted2", "Predicted3", "Actual"])
    df.plot(y=["Predicted", "Actual"])
    plt.show()


if __name__ == '__main__':
    analysis()