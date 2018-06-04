import mlp_rprop_sem_realimentacao as mlpsr
import mlp_rprop_realimentacao as mlpr
import wnn as WNN
import wnn_rprop as wnn_rprop

import matplotlib.pyplot as plt
import numpy as np

epocas = np.arange(1, 20001, 1)

### WNN
wnn = WNN.WNN()
wnn.load_function()
MSE_WNN = wnn.train()

### WNN RPROP
wnnr = wnn_rprop.WNN_RPROP()
wnnr.load_function()
MSE_WNNR = wnnr.train()

### MLP COM REALIMENTACAO
mlp_r = mlpr.MLP_R()
mlp_r.load_function()
MSE_MLP_R = mlp_r.train()

### MLP SEM REALIMENTACAO
mlp_sr = mlpsr.MLP()
mlp_sr.load_function()
MSE_MLP = mlp_sr.train()

plt.ioff()
plt.figure()
grahp_WNN, = plt.semilogy(epocas, MSE_WNN, label="grahp_WNN")
graph_WNNR, = plt.semilogy(epocas, MSE_WNNR, label="graph_WNNR")
graph_MLP_R, = plt.semilogy(epocas, MSE_MLP_R, label="graph_MLP_R")
graph_MLP, = plt.semilogy(epocas, MSE_MLP, label="graph_MLP")
plt.legend([grahp_WNN, graph_WNNR, graph_MLP_R, graph_MLP], ['WNN', 'WNN Rprop', 'MLP Feedback', 'MLP'])
plt.title('Comparative')
plt.grid(True)
plt.show()