function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% We need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta_new = theta;
theta_new(1) = 0;
reg1 = lambda/(2 * m) * (theta_new' * theta_new);
J = 1/(2*m) * sum((X * theta - y).^2) + reg1;
grad = 1/m * (X' * ((X * theta)-y)) + lambda/m*theta_new;

% Compute the cost and gradient of regularized linear 
% regression for a particular choice of theta.
%
% We should set J to the cost and grad to the gradient.

grad = grad(:);

end
