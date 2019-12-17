import copy
import os.path
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
#import theano.tensor as th

class neuron:
    # Конструктор, срабатывает при создании нейрона
    # n - количество входов нейнота, не считатя фиктивного (порога)
    #
    # save пробовать ли прочиталь сохранённые веса. Если save = True, то
    # проверяем наличие файла filename
    # Если файл не найден, то генерируем новые случайные веса
    # Если save = False, то файл даже не ищем, а сразу генерируем 
    # новые веса
    #
    # typeF - выбор логистической функции. Возможные варианты:
    # logistic, treshold
    #
    #
    #
    def __init__(self, n=1, save=False, filename = "temp.txt", typeF = "logistic"):
        if save:    # Если требуется проверить сохранение 
            if os.path.exists(filename):  # если файл с весами существует
                f = open(filename,"rt")
                lines = f.readlines()
                #print(lines)
                f.close()
                # В массиве весов сначала идут n штук весов, которые 
                # умножаются на входы, затем 
                # Последний вес умножается на фиктивный вход=1
                # для моделирования порога
                # w[0]..w[n-1] - веса, w[n] = w[-1] - порог
                self.n = len(lines)-1   # Количество весов без порога
                print("Сохренённые веса загружены в количестве", self.n, "и 1 порог!")
                self.w = list(map(float,lines))
                self.w = np.array(self.w)
                print(self.w)
                if n != self.n:
                    print("Предупрежедение! Запрошенное  число входов", n, "не совпадает с сохранённым", self.n)
            else:
                print("Генерируем новые веса!")
                self.w = np.random.sample(n+1)-0.5
                print(self.w)
                self.n = n
        else:
            print("Сразу генерируем новые веса без проверки сохранения!")
            self.w = np.random.sample(n+1)-0.5
            print(self.w)
            self.n = n
        if typeF in set(["logistic","treshold"]):
            self.typeF = typeF
            print("Нейрон использует активационную функцию", typeF)
        else:
            self.typeF = "logistic"
            print("Тип активационной функции указан неверно. Использую сигмоиду!")
        
     
    # Сохранение весов в файл
    def SaveW(self,filename = "temp.txt"):
        f = open("temp.txt","wt")
        for y in self.w:
            f.write(str(y)+'\n')
        f.close()
        
    # Логистическая функция
    def logisticFunction(self, x):
        a = 1 / (1 + np.exp(-x))
        #if a == 1:
        #    a = 0.99999  # make smallest step to the direction of zero
        #elif a == 0:
        #    a = 0.00001  # It is possible to use np.nextafter(0, 1) and
        ## make smallest step to the direction of one, but sometimes this step
        ## is too small and other algorithms fail :)
        return a

    # Пороговая функция
    def tresholdFunction(self, x):
        if x >= 0:
            return 1
        else:
            return 0

        
    # Вычисление выходного сигнала нейрона с логистической функцией
    def NeuronOutputLog(self,NeuronInput):
        if len(NeuronInput)+1 != len(self.w):
            print("Ошибка! Количество входов не равно числу весов!")
            return None
        else:
            tmp = NeuronInput
            tmp = np.append(tmp,1)
            return self.logisticFunction(sum(tmp*self.w))
        
    # Вычисление выходного сигнала нейрона с пороговой функцией
    def NeuronOutputTr(self,NeuronInput):
        if len(NeuronInput)+1 != len(self.w):
            print("Ошибка! Количество входов не равно числу весов!")
            return None
        else:
            tmp = NeuronInput
            tmp = np.append(tmp,1)
            return self.tresholdFunction(sum(tmp*self.w))
 
    # Вычисление выходного сигнала нейрона
    def NeuronOutput(self,NeuronInput):
        if len(NeuronInput)+1 != len(self.w):
            print("Ошибка! Количество входов не равно числу весов!")
            return None
        else:
            tmp = NeuronInput
            tmp = np.append(tmp,1)
            s = sum(tmp*self.w)
            if self.typeF == "logistic":
                return self.logisticFunction(s)
            else:
                return self.tresholdFunction(s)

    # Обучение нейрона
    # y - обучающая пара y[0] - входной вектор из n компонент
    # y[1] - целевое значение выхода
    def Neuron1Teach(self,y):
        if len(y[0]) != self.n:
            print("Неверная размерность обучающей пары")
        else:
            targ = y[1]
            o = self.NeuronOutput(y[0])
            delta = targ-o
            tmpx = np.append(y[0],1)
            self.w = self.w + delta*tmpx*0.01
            #print("Успешное обучение!!!")
               
    #  Печать весов нейрона
    def printW(self):
        print(self.w)
        
    
#%%
        
N = 20 # Количество отстчётов сигнала на  длительности наблюдения
       

n1 = neuron(n=N,save=True,filename = "net2.txt")
n1.SaveW()
x =  np.random.sample(n1.n)
#x = np.append(x,1)
print("Входы:")
print(x)
print("Выход:")
print(n1.NeuronOutputLog(x))
print(n1.NeuronOutputTr(x))
print(n1.NeuronOutput(x))
plt.plot(n1.w,"rx")


#%%

# Готовим обучающую конструкцию: Это список, первый элемент которого - это 
# целевое значение выхода, второй элемент - массив входов
# 0 и 1 - это амплитуды сигнала, 0 - сигнала нет, 1 - сигнал есть
m = (0,1)
#z = np.random.random()*10.0
z = 10
sigma = math.sqrt(N)/z
# Обучаем
print("Веса до обучения:")
n1.printW()

#%%

Epoch = 1000
statusbar = "-"*100
print(statusbar)
for i in range(Epoch):
    #Генерируем обучающую пару
    y = []
    R = np.random.randint(0,2)
    #z = np.random.random()*10.0
    z=10
    sigma = math.sqrt(N)/z
    y.append(np.random.normal(m[R],sigma,n1.n))
    y.append(R)
    #print("Обучающая пара:")
    #print(y)
    #print(i)
    n1.Neuron1Teach(y)
    print(i,"="*((i+1)*100//Epoch),"-"*(100-(i+1)*100//Epoch))



#%%
print("Веса после обучения:")
n1.printW()
plt.plot(n1.w)


#%%

n1.SaveW()


#%%


print("Проверка в боевом режиме")

V = 7

for i in range(V):
    #Генерируем случайную пару
    y = []
    R = np.random.randint(0,2)
    #z = np.random.random()*10.0
    z=10
    sigma = math.sqrt(N)/z
    y.append(np.random.normal(m[R],sigma,n1.n))
    y.append(R)
    #print("Проверяющая пара:")
    #print(y)
    print("Номер испытания =",i,"Истинное значение =", R, "Ответы нейрона:",
          n1.NeuronOutput(y[0]), n1.NeuronOutputTr(y[0]))

#%%
    
    
    
    

class network:
    # layers -list [5 10 10 5] - 5 input, 2 hidden
    # layers (10 neurons each), 5 output

    def create(self, layers):
        theta = [0]
        # for each layer from the first (skip zero layer!)
        for i in range(1, len(layers)):
            # create nxM+1 matrix (+bias!) with random floats in range [-1; 1]
            theta.append(
                np.mat(np.random.uniform(-1, 1, (layers[i], layers[i - 1] + 1))))
        nn = {'theta': theta, 'structure': layers}
        return nn

    def runAll(self, nn, X):
        z = [0]
        m = len(X)
        a = [copy.deepcopy(X)]  # a[0] is equal to the first input values
        logFunc = self.logisticFunctionVectorize()
        # for each layer except the input
        for i in range(1, len(nn['structure'])):
            # add bias column to the previous matrix of activation functions
            a[i - 1] = np.c_[np.ones(m), a[i - 1]]
            # for all neurons in current layer multiply corresponds neurons
            z.append(a[i - 1] * nn['theta'][i].T)
            # in previous layers by the appropriate weights and sum the
            # productions
            a.append(logFunc(z[i]))  # apply activation function for each value
        nn['z'] = z
        nn['a'] = a
        return a[len(nn['structure']) - 1]

    def run(self, nn, input):
        z = [0]
        a = []
        a.append(copy.deepcopy(input))
        a[0] = np.matrix(a[0]).T  # nx1 vector
        logFunc = self.logisticFunctionVectorize()
        for i in range(1, len(nn['structure'])):
            a[i - 1] = np.vstack(([1], a[i - 1]))
            z.append(nn['theta'][i] * a[i - 1])
            a.append(logFunc(z[i]))
        nn['z'] = z
        nn['a'] = a
        return a[len(nn['structure']) - 1]

    def logisticFunction(self, x):
        a = 1 / (1 + np.exp(-x))
        if a == 1:
            a = 0.99999  # make smallest step to the direction of zero
        elif a == 0:
            a = 0.00001  # It is possible to use np.nextafter(0, 1) and
        # make smallest step to the direction of one, but sometimes this step
        # is too small and other algorithms fail :)
        return a

    def logisticFunctionVectorize(self):
        return np.vectorize(self.logisticFunction)

    def costTotal(self, theta, nn, X, y, lamb):
        m = len(X)
        # following string is for fmin_cg computaton
        if type(theta) == np.ndarray:
            nn['theta'] = self.roll(theta, nn['structure'])
        y = np.matrix(copy.deepcopy(y))
        # feed forward to obtain output of neural network
        hAll = self.runAll(nn, X)
        cost = self.cost(hAll, y)
        # apply regularization
        return cost / m + (lamb / (2 * m)) * self.regul(nn['theta'])

    def cost(self, h, y):
        logH = np.log(h)
        log1H = np.log(1 - h)
        # transpose y for matrix multiplication
        cost = -1 * y.T * logH - (1 - y.T) * log1H
        # sum matrix of costs for each output neuron and input vector
        return cost.sum(axis=0).sum(axis=1)

    def regul(self, theta):
        reg = 0
        thetaLocal = copy.deepcopy(theta)
        for i in range(1, len(thetaLocal)):
            # delete bias connection
            thetaLocal[i] = np.delete(thetaLocal[i], 0, 1)
            # square the values because they can be negative
            thetaLocal[i] = np.power(thetaLocal[i], 2)
            # sum at first rows, than columns
            reg += thetaLocal[i].sum(axis=0).sum(axis=1)
        return reg

    def backpropagation(self, theta, nn, X, y, lamb):
        layersNumb = len(nn['structure'])
        thetaDelta = [0] * (layersNumb)
        m = len(X)
        # calculate matrix of outpit values for all input vectors X
        hLoc = copy.deepcopy(self.runAll(nn, X))
        yLoc = np.matrix(y)
        thetaLoc = copy.deepcopy(nn['theta'])
        derFunct = np.vectorize(
            lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))))

        zLoc = copy.deepcopy(nn['z'])
        aLoc = copy.deepcopy(nn['a'])
        for n in range(0, len(X)):
            delta = [0] * (layersNumb + 1)  # fill list with zeros
            # calculate delta of error of output layer
            delta[len(delta) - 1] = (hLoc[n].T - yLoc[n].T)
            for i in range(layersNumb - 1, 0, -1):
                # we can not calculate delta[0] because we don't have theta[0]
                # (and even we don't need it)
                if i > 1:
                    z = zLoc[i - 1][n]
                    # add one for correct matrix multiplication
                    z = np.c_[[[1]], z]
                    delta[i] = np.multiply(
                        thetaLoc[i].T * delta[i + 1], derFunct(z).T)
                    delta[i] = delta[i][1:]
                thetaDelta[i] = thetaDelta[i] + delta[i + 1] * aLoc[i - 1][n]

        for i in range(1, len(thetaDelta)):
            thetaDelta[i] = thetaDelta[i] / m
            thetaDelta[i][:, 1:] = thetaDelta[i][:, 1:] + \
                thetaLoc[i][:, 1:] * (lamb / m)  # regularization

        if type(theta) == np.ndarray:
            # to work also with fmin_cg
            return np.asarray(self.unroll(thetaDelta)).reshape(-1)
        return thetaDelta

    # create 1d array form lists like theta
    def unroll(self, arr):
        for i in range(0, len(arr)):
            arr[i] = np.matrix(arr[i])
            if i == 0:
                res = (arr[i]).ravel().T
            else:
                res = np.vstack((res, (arr[i]).ravel().T))
        res.shape = (1, len(res))
        return res
    # roll back 1d array to list with matrices according to given structure

    def roll(self, arr, structure):
        rolled = [arr[0]]
        shift = 1
        for i in range(1, len(structure)):
            temparr = copy.deepcopy(
                arr[shift:shift + structure[i] * (structure[i - 1] + 1)])
            temparr.shape = (structure[i], structure[i - 1] + 1)
            rolled.append(np.matrix(temparr))
            shift += structure[i] * (structure[i - 1] + 1)
        return rolled



