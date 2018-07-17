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
fprintf('***BEGINNING...');

y_matrix = zeros(num_labels)(y,:);

for temp = 1:m
   y_matrix(temp, y(temp)) = 1;
endfor

% forward prop
X = [ones(m,1) X];

% I changed this to 0 from m
for temp = 1 : m

  % a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
  a1 = X(temp,:);
  a1bad = [ones(rows(X),1), X];

  %z2 equals the product of a1 and Theta1 
  z2 = a1 * Theta1';

  % a2 is the result of passing z2 through g()
  a2Temp = sigmoid( z2 );
  
  a2bad = [ones(rows(a2Temp),1), a2Temp];
  a2 = [1, a2Temp];

  % z3 equals the product of a2  and Theta2
  % a3 is the result of passing z_3z through g()
  z3 = a2 * Theta2';
  a3 =  sigmoid(z3); 
  
  % Cost function
  %junk1 =y_matrix(temp,:);
  %junk2 =log(a3');
  
  part1 = (-1 * y_matrix(temp,:) * log(a3'));
  
  %junk3 = (1-y_matrix(temp,:));
  %junk4 = (log(1-a3'));
  
  part2 = (1-y_matrix(temp,:)) * log(1-a3');
  J = J +  (part1-part2);
 
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
  J
fprintf('Ending\n');

if (0=1)

              % Part 2: Implement the backpropagation algorithm to compute the gradients
              %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
              %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
              %         Theta2_grad, respectively. After implementing Part 2, you can check
              %         that your implementation is correct by running checkNNGradients
              %
              %         Note: The vector y passed into the function is a vector of labels
              %               containing values from 1..K. You need to map this vector into a 
              %               binary vector of 1's and 0's to be used with the neural network
              %               cost function.
              %
              %         Hint: We recommend implementing backpropagation using a for-loop
              %              ' over the training examples if you are implementing it for the 
              %               first time.
              %

              fprintf('Starting backprop');
              for temp=1:m

              % ************** forward prop again (copied)
              % a1 equals the X; input matrix with a column of 1's added (bias units) as the first column.
                a1 = X(temp,:);

                %z2 equals the product of a1 and Theta1 
                %z2 = a1 * Theta1';
                z2 = Theta1 * a1';

                % a2 is the result of passing z2 through g()
                a2 = [1; sigmoid(z2)];

                %z3 = a2 * Theta2';
                z3=Theta2 * a2;
                a3 =  sigmoid(z3); 
                
              % ********************** end copy

               % d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
                d3 = a3 - y_matrix(temp,:)'; 
                
                %d2 is tricky. It uses the (:,2:end) columns of Theta2. 
                %d2 is the product of d3 and Theta2 (without the first column), 
                %then multiplied element-wise by tdbhe sigmoid gradient of z2. 
                %The size is (m x r) \cdot⋅ (r x h) --> (m x h). The size is the same as z2.
                d2Temp = d3 * Theta2(:,2:end);
                d2 = d2Temp .* [1; sigmoidGradient(z2)];

                % Delta1 is the product of d2 and a1. The size is (h x m) \cdot⋅ (m x n) --> (h x n)
                Delta1 = d2' * a1;

                % Delta2 is the product of d3 and a2. The size is (r x m) \cdot⋅ (m x [h+1]) --> (r x [h+1])
                Delta2 = d3' * a2;

                % Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
                Theta1_grad = Theta1_grad + Delta1;
                Theta2_grad = Theta2_grad + Delta2;

                % Part 3: Implement regularization with the cost function and gradients.
                %
                %         Hint: You can implement this around the code for
                %               backpropagation. That is, you can compute the gradients for
                %               the regularization separately and then add them to Theta1_grad
                %               and Theta2_grad from Part 2.
                %
                % =========================================================================

              endfor

              fprintf('\n\n***ENDING\n\n\n\n');

              % Unroll gradients
              grad = [Theta1_grad(:) ; Theta2_grad(:)];

endif 

end

