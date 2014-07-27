# Machine learning basics

- Computer program learns from experience E wrt some task T and some performance measure P, if its performance on T as measured by P, improves with experience E.

## Supervised learning

- Give the 'right answers' to the algorithm: example data is labelled
- Regression problem: predict continuous valued output
- Classification problem: discrete valued output

## Unsupervised learning

- Dataset is not labeled
- Look for structure within data
- Clustering algorithm can be used to group similar data

# Linear regression with one variable

## Model representation

- m = number of training examples
- x = input variable/feature vector
- y = output variable/'target' variable
- h = learning algorithm (hypothesis function)
- h_theta(x) = theta_0 + theta_1 x (linear function in x)

x -> h -> y

## Cost function

- Want to choose theta_0 and theta_1 so that h_theta(x) is close to y for our training sets, x/y
- So minimise J = \sigma_{i=1}^{m}(h_theta(x_i) - y_i)^2 by changing thetas (squared error function)
- Squared error function is most common cost function for regression problems

## Gradient descent

- Used to minimise cost function J (but is a general algorithm to minimise a function)
- Algorithm:
  - Start with theta_1, theta_2
  - Find best direction to move in so that you go down hill
  - Keep changing thetas to reduce J(thetas) until we end up at minimum
- Converges to a local optima (not necessarily global)

theta_j = theta_j - \alpha* \frac{d}{d theta_j} J(theta_0, theta_1)

- Alpha is the learning rate. Large alpha means large steps (large learning rate).
- Need to simultaneously update all variables.

## Gradient descent for linear regression

- Learning rules for gradient descent in linear regession: 

   theta_0 = theta_0 - alpha * 1/m * sum(h_theta(x(i)) - y(i))

   theta_1 = theta_1 - alpha * 1/m * sum(h_theta(x(i)) - y(i))*x(i)

- Cost function for linear regression is always a convex function (bow shaped function), only one local optima. So GD always converges to global optima.
- Batch gradient descent: uses all of the training samples (ie error for all of them)
- Gradient descent scales better to large problems than the 'normal equation' (direct solution) approach







