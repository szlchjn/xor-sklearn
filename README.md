# xor-sklearn
Solving xor problem using multilayer perceptron with regression in scikit
## Problem overview
The XOr problem is a classic problem in artificial neural network research. It consists of predicting output value of **exclusive-OR** gate, using a feed-forward neural network, given truth table like the following:
####
| A| B| A ⊕ B|
| :----- | :----- | ----: |
| 0   | 0   | 0  |
| 0   | 1   |  1 |
| 1   | 0   | 1  |
| 1   | 1   | 0  |
####
Because the result space of this problem cannot be linearly separated it is considered to be basic example of the need to use multi-layered network.
### Limitations
Originally this is an obvious classification problem - although in this code, fitting this simple model of one hidden layer with two hidden neurons, to a regressor insted of a classifier yields somewhat more interesting results. With that being said, the results inside the domain should be interpreted only as an example data, because *true* results exist **only at the corners** as per truth table mentioned above.
## Results
First type of results are two different cases where the regressor finds correlation between input data on the diagonals of the domain.
![white](https://user-images.githubusercontent.com/30974121/38496671-a491f6fe-3bfe-11e8-9e4a-a73a1970c407.png)
![black](https://user-images.githubusercontent.com/30974121/38496667-a2414ada-3bfe-11e8-9656-36cea6ab4cfb.png)
### Not converging 
Radnom weight initialization can produce values which lead to our NN do not converging within max iteration threshold, thus producing this "stuck" result.
![stuck](https://user-images.githubusercontent.com/30974121/38496669-a38f0c10-3bfe-11e8-9e60-e690a85d84fb.png)
### Anomalies
Increasing the number of hidden layers and neurons in each layer, can push the network to converge faster but also causes NN to produce peculiar results.
![anomaly1](https://user-images.githubusercontent.com/30974121/38496663-a13e79be-3bfe-11e8-84b2-a0e1dea993ba.png)
