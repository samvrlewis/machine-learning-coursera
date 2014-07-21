# Neural networks : representation

- Is a non-linear classifier
- Better than linear classifier as for lots of features, for eg: if you need all quadratic features (x1x2, x2x3 etc..) then you need lots of features which can cause overfitting and is computationally expensive

## Model representation for NN
- Input wires and output wires, neuron performs computation based on inputs and produces output
- Neurons communicate via spikes
- Model neuron as logistic unit with input wires and then outputs value on output wire
- x_0 = 1 is known as the bias unit
- h(x) = 1/(1 + exp(-theta^T*x)) is sigmoid activation function
- Neural network is network of neurals in layers
- Input layer -> hidden layers -> output layer
- a_i^j = activation of unit i in layer j
- theta^j = matrix of weights controlling mapping from layer j to layer j+1


