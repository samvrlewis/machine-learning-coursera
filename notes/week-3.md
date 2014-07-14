# Logistic regression
- Used for classification problems
- Has the property than 0 <= h_theta(x) <= 1

## Classification
- Variable y is discrete variable for classification
  - Eg: spam/not spam, malignant/benign
- Binary classification {0, 1} - {0: negative class, 1: positive class}
- Can use gradient descent (which gets a cts fn) and then threshold classifier output
  - Doesn't work well though for outlying examples
  - Often isn't a good idea, can work well sometimes but not usually
  
## Hypothesis representation
- Want hypothesis that fufills 0 <= h_theta(x) <= 1
- Sigmoid function (logistic function) : g(z) = 1/(1 + exp(-z))
- Let our new hypothesis, h_theta(x) = g(h_theta(x)) = 1/(1 + exp(-theta^T*x))
- Then h_theta(x) = estimated probability that y=1 on input x, ie P(y=1|x;theta)

## Decision boundary
- If h_theta(x) > 0.5 predict y = 1, otherwise predict y=0
- For the sigmoid function, g(z) >= 0.5 for z >= 0
  - So h_theta(x) >= 0.5 when theta^T*x >= 0
  - h_theta(x) <= 0.5 when theta^T*x <= 0
  - So predict y = 1 for theta^T*x >= 0, ie theta_0 + theta_1*x1 + theta_2*x2 + ... >= 0
- Decision boundary is theta^T*x = 0

## Cost function
- Using the square cost function that was used for linear regression in logistic regression does not work well
  - Because sigmoid non linear J(theta) will be non convex (lots of local optima)
- Need a cost function that is convex:
- Cost(h_theta(x), y) = { -log(h_theta(x)) if y=1 or -log(1-h_theta(x)) if y=0)
- This penalises wrong classifications by very large costs but gives 0 cost for correct classification
- Can be simpler to write cost(h_theta(x), y) = -ylog(h_theta(x)) - (1-y)log(1-h_theta(x))
- Then J(theta) = 1/m * sum(cost(h_theta(x_i), y_i))

## Fitting parameters
- Again, need to find min(theta) J(theta)
- Use gradient descent: theta_j = theta_j - alpha* d/dtheta_j J(theta)
  - d/dtheta_j J(theta) = (h_theta(x_i) - y_i)*x_j^i
  - Identical to linear regression apart from hypothesis function being different
- Feature scaling also useful for logistic regression

## Advanced optimisation
- Other algorithms can be used to optimise J(theta)
  - Conjugate gradient
  - BFGS
  - L-BFGS
- Don't manually pick alpha
- Often faster than gradient descent
- But more complex (but possible to use without understanding it)
- Can use 'fminunc' in octave

## Multiclass classification
- One vs all
  - Consider one class then group all other classes into one class
  - Will get a hypothesis function for each class
  - So have h^i_theta(x) = P(y=i|x; theta) for each class i
  - Then for a new input, run each hypothesis and choose the one with the largest value
