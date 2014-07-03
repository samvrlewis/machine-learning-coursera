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







