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
  - Without feature scaling contours can be very skewed
- Want to get -1 <= x_i <= 1 normally (roughly, on same order of magnitude)
- Mean normalisation: replace x_i with x_i - mu_i to make features have ~0 mean

## Choosing alpha

- Plotting J(theta) helps to see if gradient descent is working
- If J(theta) is increasing or oscillating, alpha needs to be made smaller

## Features and polynomial regression

- Can create new features based on other features
  - For eg: area = frontage x depth
- Can fit other polynomials of other orders
  - Eg: h_theta(x) = theta_0 + theta_1(size) + theta_2(size)^2
- There are algorithms to choose features to use

## Normal equation
- Gradient descent is an iterative approach, normal equation is an analytical solution
- Take partials of cost function and set to 0 to find thetas that minimise cost
- Using the normal equation involves taking inverses, slow for large amounts of features


