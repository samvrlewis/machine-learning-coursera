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

