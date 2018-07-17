function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
 %   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%  
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
% 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables 
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%fprintf('***BEGINNING...');
  
y_matrix = zeros(num_labels)(y,:);

for temp = 1:m  
   y_matrix(temp, y(temp)) = 1;
endfor

% forward prop
%X = [ones(m,1) X];



% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1) X];


%z2 equals the product of a1 and Theta1 
z2 = a1 * Theta1';

% a2 is the result of passing z2 through g()


a2= [ones(m,1), sigmoid( z2 )];

% z3 equals the product of a2  and Theta2
% a3 is the result of passing z_3z through g()
z3 = a2 * Theta2';
a3 =  sigmoid(z3); 


part1 = (-1 .* y_matrix) * log(a3');
part2 = (1 .- y_matrix) * log(1 .- a3');

for diag1 = 1: size(part1)(1)
    diagCost = part1(diag1,diag1) - part2(diag1, diag1);
    J = J + diagCost;
endfor

J = J/m;

%regularization now
%lambda
%m
sumTheta1 = sum(sum( (Theta1(:,2:end).^2)));
sumTheta2 = sum(sum( (Theta2(:,2:end).^2)));
regularized = (lambda/(2*m))* (sumTheta1 + sumTheta2);

%regularized
J = J+ regularized;
%J
%fprintf('Ending\n');

%backprop

d3 = a3 - y_matrix; 
z2 = a1 * Theta1';
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
delta1 = d2' * a1;
delta2 = d3' * a2;


Theta1_grad = ((1/m)*delta1) + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = ((1/m)*delta2) + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

