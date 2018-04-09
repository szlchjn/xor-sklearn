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
First type of results are two different cases where the regressor finds corelation between input data on the diagonals of the domain.

### Not converging 
Radnom weight initialization can produce values which lead to our NN do not converging within max iteration threshold, thus producing this "stuck" result.

### Anomalies
Increasing the number of hidden layers and neurons in each layer, can push the network to converge faster but also causes NN to produce anomalies.
