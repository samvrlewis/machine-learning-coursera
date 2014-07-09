# Linear regression with multiple features

- Used for modeling problems with multiple variables

## Notation

- Xi is input feature x. y is output variable. 
- n is number of features
- x_j^i is value of feature j in ith training example

## Linear regression hypothesis in multiple features

- h_theta(x) = theta_0 + theta_1*x_1 + theta_2*x_2 + ... + theta_n*x_n
- For convenience, define x_0 = 1 so that h_theta(x) = theta^T x

## Gradient descent for multiple variables

- Think of theta as a n+1 dimensional vector
- Then J(theta) = 1/2m * sum((h_theta(x(i)) - y(i))^2)
- Update rule theta_j = theta_j - alpha*1/m * sum((h_theta(x(i)) - y(i))^2*x_j(i))

## Feature scaling

- Make features on a similar scale
- Makes gradient descent converge quicker
.. - Without feature scaling contours can be very skewed
- Want to get -1 <= x_i <= 1 normally (roughly, on same order of magnitude)
- Mean normalisation: replace x_i with x_i - mu_i to make features have ~0 mean