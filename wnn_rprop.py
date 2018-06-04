import matplotlib.pyplot as plt  # Para plotar graficos
import numpy as np  # Array do Python
from math  import sqrt, pi

class WNN_RPROP(object):
    def __init__(self, delta0=0.008, delta_max=0.01, delta_min=1e-4, epoch_max=20000, Ni=1, Nh=40, Ns=1):
        ### Inicializando parametros
        self.delta0 = delta0
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_plus = 1.2
        self.eta_less = 0.5
        self.epoch_max = epoch_max
        self.Ni = Ni
        self.Nh = Nh
        self.Ns = Ns
        self.Aini = 0.1

    def load_function(self):
        x = np.arange(-6, 6, 0.2)
        self.N = x.shape[0]
        xmax = np.max(x)

        self.X_train = x / xmax
        self.d = 1 / (1 + np.exp(-1 * x))*(np.cos(x) - np.sin(x))

    def sig_dev2(self, theta):
        return 2*(1 / (1 + np.exp(-theta)))**3 - 3*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))

    def sig_dev3(self, theta):
        return -6*(1 / (1 + np.exp(-theta)))**4 + 12*(1 / (1 + np.exp(-theta)))**3 - 7*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))

    def sig_dev4(self, theta):
        return 24*(1 / (1 + np.exp(-theta)))**5 - 60*(1 / (1 + np.exp(-theta)))**4 + 50*(1 / (1 + np.exp(-theta)))**3 - 15*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))
    
    def sig_dev5(self, theta):
        return -120*(1 / (1 + np.exp(-theta)))**6 + 360*(1 / (1 + np.exp(-theta)))**5 - 390*(1 / (1 + np.exp(-theta)))**4 + 180*(1 / (1 + np.exp(-theta)))**3 - 31*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))
    
    def train(self):
        ### Inicializando os pesos
        self.A = np.random.rand(self.Ns, self.Nh) * self.Aini

        ### Inicializando os centros
        self.t = np.zeros((1, self.Nh))

        idx = np.random.permutation(self.Nh)
        for j in xrange(self.Nh):
            self.t[0,j] = self.d[idx[j]]
        
        ### Inicializando as larguras
        self.R = abs(np.max(self.t) - np.min(self.t)) / 2

        self.tat = np.ones(self.t.shape) * self.delta0
        self.taA = np.ones(self.A.shape) * self.delta0
        grt_ant = grR_ant = grA_ant = 0

        MSE = np.zeros(self.epoch_max)
        plt.ion()

        for epoca in xrange(self.epoch_max):
            z = np.zeros(self.N)
            E = np.zeros(self.N)
            deltat = np.ones(self.t.shape)
            deltaA = np.ones(self.A.shape)
            gradt = gradR = gradA = 0

            index = np.random.permutation(self.N)

            for i in index:
                xi = self.X_train[i]#np.array([self.X_train[i]]).reshape(1, -1)
                theta = (xi - self.t) / self.R
                yj = self.sig_dev2(theta)
                z[i] = np.dot(self.A, yj.T)[0][0]

                e = self.d[i] - z[i]
                #self.A = self.A + (self.eta * e * yj)
                #self.t = self.t - (self.eta * e * self.A / self.R * self.sig_dev3(theta))
                #self.R = self.R - (((self.eta * e * self.A * (xi - self.t)) / self.R**2) * self.sig_dev3(theta))
                gradA += (-e * yj)
                gradt += (e * self.A / self.R * self.sig_dev3(theta))
                self.R = self.R - (self.delta0 * (e * self.A * (xi - self.t)) / self.R**2) * self.sig_dev3(theta)
                
                #self.R -= (self.delta0 * gradR)

                E[i] = 0.5 * e**2

            grt = np.sign(gradt)
            grA = np.sign(gradA)

            if epoca == 0:
                self.t += (-self.delta0 * gradt)
                self.R += (-self.delta0 * gradR)
                self.A += (-self.delta0 * gradA)
            else:
                Dt = grt * grt_ant
                DA = grA * grA_ant

                sizet = Dt.shape
                sizeA = DA.shape

                for i in xrange(sizet[0]):
                    for j in xrange(sizet[1]):
                        if (Dt[i,j] > 0):
                            self.tat[i,j] = min(self.tat[i,j] * self.eta_plus, self.delta_max)
                        elif (Dt[i,j] < 0):
                            self.tat[i,j] = max(self.tat[i,j] * self.eta_less, self.delta_min)

                        if (grt[i,j] > 0):
                            deltat[i,j] = -self.tat[i,j]
                        elif (grt[i,j] < 0):
                            deltat[i,j] = self.tat[i,j]
                
                for i in xrange(sizeA[0]):
                    for j in xrange(sizeA[1]):
                        if (DA[i,j] > 0):
                            self.taA[i,j] = min(self.taA[i,j] * self.eta_plus, self.delta_max)
                        elif (DA[i,j] < 0):
                            self.taA[i,j] = max(self.taA[i,j] * self.eta_less, self.delta_min)

                        if (grA[i,j] > 0):
                            deltaA[i,j] = -self.taA[i,j]
                        elif (grA[i,j] < 0):
                            deltaA[i,j] = self.taA[i,j]

                self.t += deltat
                self.A += deltaA
            
            grt_ant = grt
            grA_ant = grA

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
        d, = plt.plot(self.X_train, self.d, '.', label="d")
        plt.legend([y, d], ['WNN Output', 'Desired Value'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Wavelet Neural Network - Rprop')
        plt.text(np.min(self.X_train) - np.max(self.X_train) * 0.17  , np.min(self.d) - np.max(self.d) * 0.17, 'Progress: ' + str(round(float(epoca) / self.epoch_max * 100, 2)) + '%')
        plt.axis([np.min(self.X_train) - np.max(self.X_train) * 0.2, np.max(self.X_train) * 1.2, np.min(self.d) - np.max(self.d) * 0.2, np.max(self.d) * 1.5])
        plt.show()
        plt.pause(1e-100)

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
        plt.title('Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(self.X_train, self.d)
        plt.show()

#wnn = WNN_RPROP()

#wnn.load_function()
#MSE_WNN = wnn.train()
#wnn.plot_MSE(MSE_WNN)

### Generico
"""import matplotlib.pyplot as plt  # Para plotar graficos
import numpy as np  # Array do Python
from math  import sqrt, pi

class WNN(object):
    def __init__(self, delta0=0.1, delta_max=50, delta_min=1e-6, epoch_max=10000, Ni=2, Nh=40, Ns=1):
        ### Inicializando parametros
        self.delta0 = delta0
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_plus = 1.2
        self.eta_less = 0.5
        self.epoch_max = epoch_max
        self.Ni = Ni
        self.Nh = Nh
        self.Ns = Ns
        self.Aini = 0.1

    def load_function(self):
        x = np.arange(-6, 6, 0.2)
        self.N = x.shape[0]
        xmax = np.max(x)

        self.X_train = x / xmax
        self.d = 1 / (1 + np.exp(-1 * x))*(np.cos(x) - np.sin(x))

    def sig_dev2(self, theta):
        return 2*(1 / (1 + np.exp(-theta)))**3 - 3*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))

    def sig_dev3(self, theta):
        return -6*(1 / (1 + np.exp(-theta)))**4 + 12*(1 / (1 + np.exp(-theta)))**3 - 7*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))

    def sig_dev4(self, theta):
        return 24*(1 / (1 + np.exp(-theta)))**5 - 60*(1 / (1 + np.exp(-theta)))**4 + 50*(1 / (1 + np.exp(-theta)))**3 - 15*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))
    
    def sig_dev5(self, theta):
        return -120*(1 / (1 + np.exp(-theta)))**6 + 360*(1 / (1 + np.exp(-theta)))**5 - 390*(1 / (1 + np.exp(-theta)))**4 + 180*(1 / (1 + np.exp(-theta)))**3 - 31*(1 / (1 + np.exp(-theta)))**2 + (1 / (1 + np.exp(-theta)))
    
    def train(self):
        ### Inicializando os pesos
        self.A = np.random.rand(self.Ns, self.Nh) * self.Aini

        ### Inicializando os centros
        self.t = np.zeros((1, self.Nh))

        idx = np.random.permutation(self.Nh)
        for j in xrange(self.Nh):
            self.t[0,j] = self.d[idx[j]]
        
        ### Inicializando as larguras
        self.R = abs(np.max(self.t) - np.min(self.t)) / 2

        self.tat = np.ones(self.t.shape) * self.delta0
        self.taR = np.ones(self.R.shape) * self.delta0
        self.taA = np.ones(self.A.shape) * self.delta0
        grt_ant = grR_ant = grA_ant = 0

        MSE = np.zeros(self.epoch_max)
        plt.ion()

        for epoca in xrange(self.epoch_max):
            z = np.zeros(self.N)
            E = np.zeros(self.N)
            deltat = np.ones(self.t.shape)
            deltaR = np.ones(self.R.shape)
            deltaA = np.ones(self.A.shape)
            gradt = gradR = gradA = 0

            index = np.random.permutation(self.N)

            for i in index:
                xi = self.X_train[i]#np.array([self.X_train[i]]).reshape(1, -1)
                theta = (xi - self.t) / self.R
                yj = self.sig_dev2(theta)
                z[i] = np.dot(self.A, yj.T)[0][0]

                e = self.d[i] - z[i]
                #self.A = self.A + (self.eta * e * yj)
                #self.t = self.t - (self.eta * e * self.A / self.R * self.sig_dev3(theta))
                #self.R = self.R - (((self.eta * e * self.A * (xi - self.t)) / self.R**2) * self.sig_dev3(theta))
                gradA = (e * yj)
                gradt = (e * self.A / self.R * self.sig_dev3(theta))
                gradR = ((e * self.A * (xi - self.t)) / self.R**2) * self.sig_dev3(theta)

                E[i] = 0.5 * e**2

            grt = np.sign(gradt)
            grR = np.sign(gradR)
            grA = np.sign(gradA)

            if epoca == 0:
                self.t += (-self.delta0 * gradt)
                self.R += (-self.delta0 * gradR)
                self.A += (-self.delta0 * gradA)
            else:
                Dt = grt * grt_ant
                DR = grR * grR_ant
                DA = grA * grA_ant

                sizet = Dt.shape
                sizeR = DR.shape
                sizeA = DA.shape

                for i in xrange(sizet[0]):
                    for j in xrange(sizet[1]):
                        if (Dt[i,j] > 0):
                            self.tat[i,j] = min(self.tat[i,j] * self.eta_plus, self.delta_max)
                        elif (Dt[i,j] < 0):
                            self.tat[i,j] = max(self.tat[i,j] * self.eta_less, self.delta_min)

                        if (grt[i,j] > 0):
                            deltat[i,j] = -self.tat[i,j]
                        elif (grt[i,j] < 0):
                            deltat[i,j] = self.tat[i,j]
                
                for i in xrange(sizeR[0]):
                    for j in xrange(sizeR[1]):
                        if (DR[i,j] > 0):
                            self.taR[i,j] = min(self.taR[i,j] * self.eta_plus, self.delta_max)
                        elif (DR[i,j] < 0):
                            self.taR[i,j] = max(self.taR[i,j] * self.eta_less, self.delta_min)

                        if (grR[i,j] > 0):
                            deltaR[i,j] = -self.taR[i,j]
                        elif (grR[i,j] < 0):
                            deltaR[i,j] = self.taR[i,j]
                
                for i in xrange(sizeA[0]):
                    for j in xrange(sizeA[1]):
                        if (DA[i,j] > 0):
                            self.taA[i,j] = min(self.taA[i,j] * self.eta_plus, self.delta_max)
                        elif (DA[i,j] < 0):
                            self.taA[i,j] = max(self.taA[i,j] * self.eta_less, self.delta_min)

                        if (grA[i,j] > 0):
                            deltaA[i,j] = -self.taA[i,j]
                        elif (grA[i,j] < 0):
                            deltaA[i,j] = self.taA[i,j]

                self.t += deltat
                self.R += deltaR
                self.A += deltaA
            
            grt_ant = grt
            grR_ant = grR
            grA_ant = grA

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
        d, = plt.plot(self.X_train, self.d, '.', label="d")
        plt.legend([y, d], ['WNN Output', 'Desired Value'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Wavelet Neural Network')
        plt.text(np.min(self.X_train) - np.max(self.X_train) * 0.17  , np.min(self.d) - np.max(self.d) * 0.17, 'Progress: ' + str(round(float(epoca) / self.epoch_max * 100, 2)) + '%')
        plt.axis([np.min(self.X_train) - np.max(self.X_train) * 0.2, np.max(self.X_train) * 1.2, np.min(self.d) - np.max(self.d) * 0.2, np.max(self.d) * 1.5])
        plt.show()
        plt.pause(1e-100)

    def plot_MSE(self, MSE):
        plt.ioff()
        plt.figure(1)
        plt.title('Mean Square Error (MSE)')
        plt.xlabel('Training Epochs')
        plt.ylabel('MSE')
        plt.plot(np.arange(0, MSE.size), MSE)
        plt.show()

    def show_function(self):
        plt.figure(0)
        plt.title('Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(self.X_train, self.d)
        plt.show()

wnn = WNN()

wnn.load_function()
WNN = wnn.train()"""