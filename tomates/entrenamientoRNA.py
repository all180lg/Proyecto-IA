import neurolab as nl
import numpy as np
import scipy as sp

#datos de entrada para el entrenamiento
datos = np.matrix(sp.genfromtxt("datos.csv", delimiter=" "))
#datos de entrada a la neurona
entrada = datos[:,:-3]
#datos de salida de la neurona
objetivo = datos[:,-3:]
#max min para cada dato de entrada a la neurona 
maxmin = np.matrix([[-5, 5] for i in range(len(entrada[1,:].T))])
# valores para las capas de la neurona 
kInicial = entrada.shape[0]
k1 = int(kInicial*0.5)
k2 = int(k1*0.5)
k3 = int(k2*5)
# Crear red neuronal con 5 capas 1 de entrada 3 ocultas y 1 de salida 
rna = nl.net.newff(maxmin,[kInicial,k1,k2,k3,1])
#Cambio de algoritmo a back progation simple
rna.trainf = nl.train.train_gd
rna.train(entrada, objetivo, epochs=10000000, show=100, goal=0.03, lr=0.01)
rna.save("verdes.rna")