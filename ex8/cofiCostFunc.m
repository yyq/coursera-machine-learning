function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

matrics = X * Theta' - Y;
matrics2 = matrics .* matrics;

J = 1 / 2 * sum(sum(matrics2 .* R));

J_reg = lambda / 2 * ( sum(sum(Theta .* Theta)));
J_reg = J_reg + lambda / 2 * (sum(sum(X .* X)));
J = J + J_reg;

for i=1:num_movies
    % For the i-th movie, get all the user index who rated? eg (1 3 5 7)
    idx = find(R(i, :) == 1);
    % Get all theta of those users,  4 * n_f
    thetatemp = Theta(idx, :);
    % For the i-th movie, get real rating of those users, 1 * 4
    ytemp = Y(i, idx);
    % 1*nf      =  1*n_f  * n_f*4        1*4       4*nf 
    X_grad(i,:) = (X(i,:) * thetatemp' - ytemp) * thetatemp; 
    X_grad(i,:) = X_grad(i,:) + lambda * X(i,:);
end

for j=1:num_users
    % For the j-th user, get all movies' index he rated, eg(1 3 5 7)
    idx = find(R(:,j) == 1);
    % Get all features of those movies,   4 * n_f
    xtemp = X( idx, :);
    % For the j-th user, get real rating of those movies, 4 * 1
    ytemp = Y(idx, j);
    %   1*n_f       = ( 4*n_f  *   n_f*1 )    4*1     
    Theta_grad(j,:) = (xtemp * Theta(j,:)' - ytemp )' * xtemp;
    Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:) ;
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
