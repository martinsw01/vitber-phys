import numpy as np

R = 100
h = 4*R/(3*np.pi)
sigma0 = 1000
sigma = 500
beta = 1 #Erstatt med kode for å finne beta

yM0 = R * np.cos(beta/2)
yC0 = yM0 - h
yMB0 = 4*R*np.sin(beta/2)**3/(3*(beta - np.sin(beta)))
yB0 = yM0 - yMB0
yD0 = yM0 - R