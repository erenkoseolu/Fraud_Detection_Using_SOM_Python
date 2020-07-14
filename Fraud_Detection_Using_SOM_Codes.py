import numpy as np
import pandas as pd

#imorting the data and converting them into array
data=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Data_Mining\\Credit_Card_Applications.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x=sc.fit_transform(x)
x

#Importing MiniSom class
from minisom import MiniSom
som=MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)

som.random_weights_init(data=x)
som.train_random(data=x, num_iteration=100)

#Creating the map
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

#Creating symbols to find the potential cheaters
for i,j in enumerate(x):
    w = som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor= colors[y[i]],
         markersize=10,
         markeredgewidth=2,
         markerfacecolor = 'None'
    )
show()

#Getting the winning node values 
mappings = som.win_map(x)
mappings
 
#The most potential cheaters have been concatenated.
frauds = np.concatenate((mappings[(8,1)], mappings[(7,5)]), axis = 0)

#The features have been inversed 
inverse_scaled_frauds=sc.inverse_transform(frauds)

#Here, the potential cheaters' IDs have been printed
print("Cheaters")
for i in inverse_scaled_frauds[:,0]:
    print(i)





















