function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% if runTraining = 1, then the training algorithms will run
runTraining = 0;

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
minError=1000000.0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Training Results:
% Final C= 1.000000 sigma= 0.100000

if (runTraining)

  cVector = [0.01 0.03, 0.1, 0.3, 1, 3, 10, 30];
  sigmaVector = [0.01 0.03, 0.1, 0.3, 1, 3, 10, 30];

  for aC = cVector
    for aSigma = sigmaVector
       model= svmTrain(X, y, aC, @(x1, x2) gaussianKernel(x1, x2, aSigma));
       predictions = svmPredict(model, Xval);
       predError = mean(double(predictions ~= yval));
       %printf("(%f, %f) => %f ", aC, aSigma, predError); 
       if ( predError <= minError )
          C = aC;
          sigma= aSigma;
          minError = predError;
        %  printf("  ******* New Minimum\n");
       else
        %  printf("    x \n");
       endif
    endfor
  endfor


  printf("\n\n=========>>>>>>>>>>>>>  Final C= %f sigma= %f\n\n", C, sigma);

endif

% =========================================================================

end
