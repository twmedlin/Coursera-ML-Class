function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
redPart = (-1 * y )' * (log(sigmoid(X*theta)));
 

bluePart =( 1 - y)' * log( 1 - sigmoid(X*theta));

unregularizedPart = (1/m) * (redPart - bluePart);


% theta(1)=0;
% regularizedPart = (lambda / (2*m)) * (theta * theta');
regularizedPart = 0;

J = unregularizedPart + regularizedPart;

% The left-side term is the vector product of X and (h - y), scaled by 1/m.
grad = ( sigmoid(X * theta) - y)' * X;
grad = (1/m) * grad; 
junk = grad * 313;


% =============================================================

end
