from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

regr = MLPRegressor(hidden_layer_sizes=(2),
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
cax = plt.imshow(np.array(output), interpolation='nearest', vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0, 1])
cbar.ax.tick_params(labelsize=15)
plt.set_cmap('gray')
plt.axis('off')

table = {'0, 0':(-2, -1),
         '0, 1':(-2, res+2),
         '1, 0':(res-2, -1),
         '1, 1':(res-2, res+2)}
for text, corner in table.items():
    ax.annotate(text, xy=corner, size=15, annotation_clip=False)

plt.show()
