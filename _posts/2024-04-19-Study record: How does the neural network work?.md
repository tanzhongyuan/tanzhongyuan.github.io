# Study record: How does the neural network work?

In fastai tutorial "How to Understand How Neural Networks Really Work," Jeremy Howard details the basic components and workings of neural networks.  

Starting with a simple definition of neural networks, the paper explains how networks make predictions by learning patterns in input data. 
The concepts of forward propagation and back propagation, as well as the role of loss functions and the importance of optimization algorithms such as gradient descent in network training are emphasized. 
In addition, Howard uses code examples to show how these concepts work in practice to help us intuitively understand how neural networks work.

## 1. Optimization algotithm: Gradient decent method

### 1.1 Concept
Gradient descent is a first-order optimization algorithm. To find the local minimum of a function using gradient descent, the function must be iteratively searched for a specified step distance point in the opposite direction of the previous point corresponding to the gradient (or approximate gradient).



Gradient descent is a kind of iterative method that can be used to solve linear and nonlinear least squares problems. When solving the model parameters of machine learning algorithms such as unconstrained optimization problems, gradient descent and least squares are the most commonly used methods.
