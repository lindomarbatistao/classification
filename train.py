from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as ap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Teste:
    def __init__(self, x_ts=0.2, x_rs=0, X_train=0, X_test=0, y_train=0, y_test=0, model=0):
        self.x_ts = x_ts
        self.x_rs = x_rs
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.menu()

    def menu(self):
        while True:
            print('\n########## Menu ############')
            print(f'[0] - Exit\n'
                  '[1] - Test Size\n'
                  '[2] - Random State\n'
                  '[3] - Test Size and Random State\n'
                  )
            op = int(input('Select option: '))
            if op == 0:
                break
            elif op == 1:
                self.test_size()
            elif op == 2:
                self.random_state()
            elif op == 3:
                self.test_size_random_state()
            else:
                print('Select valid option ...')

    def test_size(self):
        while True:
            self.x_ts = float(input('Select percent (10-60): '))/100
            if self.x_ts > 0.10 or self.x_ts < 0.60:
                self.principal()
                break
            else:
                print('Select valid option...')

    def random_state(self):
        while True:
            self.x_rs = int(input('Select option random (0-20): '))
            if self.x_rs>0 and self.x_rs<20:
                self.principal()
                break
            else:
                print('Select valid option...')

    def test_size_random_state(self):
        while True:
            self.x_ts = float(input('Select percent (10-60): ')) / 100
            if self.x_ts < 0.10 or self.x_ts > 0.60:
                print('Select valid option...')
            self.x_rs = int(input('Select option random (0-20): '))
            if self.x_rs < 0 and self.x_rs > 20:
                print('Select valid option...')
            self.principal()

    def principal(self):
        # fetch dataset
        wine = fetch_ucirepo(id=109)

        # data (as pandas dataframes)
        X = wine.data.features
        y = wine.data.targets

        # metadata
        # print(wine.metadata)

        # variable information
        # print(wine.variables)

        # Criando o Dataframe
        g = pd.DataFrame(X)
        h = pd.DataFrame(y)
        pd.set_option('future.no_silent_downcasting', True)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', 20)

        dataX = g
        #dataX.to_excel('C:/Users/lintelecom/Desktop/result/dataX.xlsx')
        # print(dataX)

        dataY = h
        # dataY.to_excel('C:/Users/lintelecom/Desktop/result/dataY.xlsx')
        # print(dataY)

        # #Transformando em vetor
        i = pd.DataFrame(dataX)
        dataX = i.values
        # print(dataX)
        j = pd.DataFrame(dataY)
        dataY = j.values
        # print(dataY)

        # Dividir os dados em conjuntos de treino e teste (por exemplo, 80% de treino e 20% de teste)
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=self.x_ts, random_state=self.x_rs, stratify=dataY)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Criar o classificador de árvore de decisão
        self.model = DecisionTreeClassifier(random_state=self.x_rs, criterion='entropy')
        self.model.fit(X_train, y_train)  # Treinar classificador

        # Faça as previsões usando o modelo
        predictions = self.model.predict(X_test)

        # Calcule a acurácia
        accuracy = accuracy_score(y_test, predictions)

        print('\n########### Training ###########')
        print(f'Percentage: {self.x_ts*100}%')
        print(f'Random: {self.x_rs}')
        print(f'Training set size: {X_train.shape}')
        print(f"Accuracy: {accuracy * 100:.2f}%")

        while True:
            print(f'\n########### Imprimir ###########\n'
                  '[0] - Exit\n'
                  '[1] - Confusion Matrix\n'
                  '[2] - Decision Tree\n'
                  )

            op = int(input('Select option: '))
            if op == 0:
                break
            elif op == 1:
                self.mapaConfusao()
            elif op == 2:
                self.arvoreDecisao()
            else:
                print('Select valid option...')

    def mapaConfusao(self):
        # Construindo o modelo
        clf = LogisticRegression(random_state=self.x_rs, solver='sag').fit(self.X_train, self.y_train)
        pred_clf = clf.predict(self.X_test)

        # Matriz de confusão
        cf_matrix = confusion_matrix(self.y_test, pred_clf)

        # Criando o heatmap
        sns.heatmap(cf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
        plt.show()

    def arvoreDecisao(self):
        wine = fetch_ucirepo(id=109)
        plt.figure(figsize=(10, 10))
        plot_tree(self.model, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
        plt.show()

        # Erros de classificação
        treino_acuracia = self.model.score(self.X_train, self.y_train)
        teste_acuracia = self.model.score(self.X_test, self.y_test)
        print(f'Classification errors in training: {1 - treino_acuracia:.2f}')
        print(f'Classification errors in test: {1 - teste_acuracia:.2f}')


