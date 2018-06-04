import matplotlib.pyplot as plt  # Para plotar graficos
import numpy as np  # Array do Python
from math import sqrt, pi

class MLP(object):
    def __init__(self, delta0=0.1, delta_max=50, delta_min=1e-6, epoch_max=20000, Ni=2, Nh=15, Ns=1):
        self.delta0 = delta0
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_plus = 1.2
        self.eta_less = 0.5
        self.epoch_max = epoch_max
        self.Ni = Ni
        self.Nh = Nh
        self.Ns = Ns
        self.Wini = 0.1

    def load_function(self):
        x = np.arange(-6, 6, 0.2)
        self.N = x.shape[0]
        xmax = np.max(x)

        self.X_train = x / xmax
        self.d = 1 / (1 + np.exp(-1 * x)) * (np.cos(x) - np.sin(x))
    
    def train(self):
        self.Wji = np.random.rand(self.Nh, self.Ni) * self.Wini
        self.Wkj = np.random.rand(self.Ns, self.Nh + 1) * self.Wini
        self.taWji = np.ones((self.Nh, self.Ni)) * self.delta0
        self.taWkj = np.ones((self.Ns, self.Nh + 1)) * self.delta0
        grji_ant = grkj_ant = 0

        MSE = np.zeros(self.epoch_max)
        plt.ion()

        for epoca in xrange(self.epoch_max):
            gradji = gradkj = 0
            deltaji = np.zeros(self.Wji.shape)
            deltakj = np.zeros(self.Wkj.shape)
            z = np.zeros(self.N)
            E = np.zeros(self.N)

            for i in xrange(self.N):
                xi = np.array([-1, self.X_train[i]]).reshape(1, -1)
                netj = np.dot(self.Wji, xi.T)
                #yj = np.tanh(netj.T)
                yj = 1 / (1 + np.exp(-netj.T))
                yj_pol = np.insert(yj[0], 0, -1).reshape(1, -1)
                z[i] = np.dot(self.Wkj, yj_pol.T)[0][0]

                e = self.d[i] - z[i]

                #gradji += np.dot((-e * self.Wkj[:,1:] * (1 - yj**2)).T, xi)
                gradji += np.dot((-e * self.Wkj[:,1:] * yj * (1 - yj)).T, xi)
                gradkj += (-e * yj_pol)

                E[i] = 0.5 * e**2

            grji = np.sign(gradji)
            grkj = np.sign(gradkj)

            if epoca == 0:
                self.Wkj += (-self.delta0 * gradkj)
                self.Wji += (-self.delta0 * gradji)
            else:
                Dji = grji * grji_ant
                Dkj = grkj * grkj_ant

                sizeji = Dji.shape
                sizekj = Dkj.shape

                for i in xrange(sizeji[0]):
                    for j in xrange(sizeji[1]):
                        if (Dji[i,j] > 0):
                            self.taWji[i,j] = min(self.taWji[i,j] * self.eta_plus, self.delta_max)
                        elif (Dji[i,j] < 0):
                            self.taWji[i,j] = max(self.taWji[i,j] * self.eta_less, self.delta_min)

                        if (grji[i,j] > 0):
                            deltaji[i,j] = -self.taWji[i,j]
                        elif (grji[i,j] < 0):
                            deltaji[i,j] = self.taWji[i,j]
                
                for i in xrange(sizekj[0]):
                    for j in xrange(sizekj[1]):
                        if (Dkj[i,j] > 0):
                            self.taWkj[i,j] = min(self.taWkj[i,j] * self.eta_plus, self.delta_max)
                        elif (Dkj[i,j] < 0):
                            self.taWkj[i,j] = max(self.taWkj[i,j] * self.eta_less, self.delta_min)

                        if (grkj[i,j] > 0):
                            deltakj[i,j] = -self.taWkj[i,j]
                        elif (grkj[i,j] < 0):
                            deltakj[i,j] = self.taWkj[i,j]

                self.Wkj += deltakj
                self.Wji += deltaji
            
            grji_ant = grji
            grkj_ant = grkj

            MSE[epoca] = np.sum(E) / self.N

            if (epoca % 200 == 0 or epoca == self.epoch_max - 1):
                if (epoca != 0):
                    plt.cla()
                    plt.clf()
                
                self.plot(z, epoca)
        
        print MSE[-1]
        
        return MSE

    def plot(self, saida, epoca):
        plt.figure(0)
        y, = plt.plot(self.X_train, saida, label="y")
        d, = plt.plot(self.X_train, self.d, '*', label="d")
        plt.legend([y, d], ['Output of Network Neural', 'Desired Value'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('MLP Rprop')
        plt.text(np.min(self.X_train) - np.max(self.X_train) * 0.17  , np.min(self.d) - np.max(self.d) * 0.17, 'Progress: ' + str(round(float(epoca) / self.epoch_max * 100, 2)) + '%')
        plt.axis([np.min(self.X_train) - np.max(self.X_train) * 0.2, np.max(self.X_train) * 1.2, np.min(self.d) - np.max(self.d) * 0.2, np.max(self.d) * 1.5])
        plt.show()
        plt.pause(1e-10)

    def plot_MSE(self, MSE):
        plt.ioff()
        plt.figure(1)
        plt.title('Mean Square Error (MSE)')
        plt.xlabel('Training Epochs')
        plt.ylabel('MSE')
        plt.semilogy(np.arange(0, MSE.size), MSE)
        plt.show()

    def show_function(self):
        plt.figure(0)
        plt.title('Second Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(self.X_train, self.d)
        plt.show()

#rprop = Rprop()

#rprop.load_function()
#MSE = rprop.train()
#rprop.plot_MSE(MSE)