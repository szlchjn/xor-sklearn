from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

regr = MLPRegressor(hidden_layer_sizes=(4),
                   activation='tanh',
                   solver='lbfgs')

model = regr.fit(X, y)

res = 50
output = [None] * res

for i in range(res):
    output[i] = [None] * res
    for j in range(res):
        x = np.array([i/res, j/res]).reshape(1, -1)
        output[i][j] = model.predict(x)[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.imshow(np.array(output), interpolation='nearest')
plt.set_cmap('gray')
plt.colorbar(orientation='vertical')
plt.show()