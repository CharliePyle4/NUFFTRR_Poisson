function [C, D] = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
% Compute the integrals C and D for a nonuniform mesh in the radial
% direction, using either Simpson's rule or trapezoidal rule.

M = length(r_m);
N = length(f_fourier_coeff(:,1)) - 1;

% Declare matrices
C = zeros(N/2+1,M-1);
D = zeros(N/2+1,M-1);

% Create the individual mesh widths.
delta = zeros(M-1,1);
for i = 1:M-1
    delta(i) = r_m(i+1) - r_m(i);
end

% Much of the index notation below is based around the
% fact that the array indexing is always positive, whereas
% the mathematics uses negative indexing.

% Trapezoidal Rule %
%------------------%

if quad_rule == 1
    for i = 1 : M-1
        for n = 1 : N/2
            C(n,i) = delta(i) / (4*(-N/2 + n -1)) ...
                * ( r_m(i)*(r_m(i)/r_m(i+1))^(-(-N/2 + n -1)) * f_fourier_coeff(n,i) ...
                + r_m(i+1) * f_fourier_coeff(n,i+1) );
            D(n+1,i) = -( delta(i) / (4*n) ) ...
                * ( r_m(i+1)*(r_m(i)/r_m(i+1))^n * f_fourier_coeff(n+N/2+1,i+1) ...
                + r_m(i)*f_fourier_coeff(n+N/2+1,i) );
        end
        C(N/2+1,i) = delta(i)/2 * ( r_m(i)*f_fourier_coeff(N/2+1,i) ...
            + r_m(i+1)*f_fourier_coeff(N/2+1,i+1) );
        
        % We must compute D(1,1) separately, since there is a computation of
        % 0*log(0) in the algorithm. That is why we exclude 'i=1' from the
        % computation.
        if i ~= 1
            D(1,i) = delta(i)/2 * ( r_m(i+1)*log(r_m(i+1))*f_fourier_coeff(N/2+1,i+1) + ...
                r_m(i)*log(r_m(i))*f_fourier_coeff(N/2+1,i) );
        end
    end
    D(1,1) = delta(1)/2*( r_m(2)*log(r_m(2))*f_fourier_coeff(N/2+1,2));
    
% With the Simpson's rule, we alot the first column of C 
% and D to contain the values of C^(1,2) and D^(M-1,M).

% Simpson's Rule %
%----------------%
elseif quad_rule == 2
    for i = 2 : M-1
        % Need to use a nonuniform Simpson's rule.
        r_temp = [r_m(i-1); r_m(i); r_m(i+1)];
        
        for n = 1 : N/2
            f_temp = [r_m(i-1)/(2*(-N/2+n-1)) * (r_m(i+1)/r_m(i-1))^(-N/2+n-1) * f_fourier_coeff(n,i-1); ...
                      r_m(i)/(2*(-N/2+n-1)) * (r_m(i+1)/r_m(i))^(-N/2+n-1) * f_fourier_coeff(n,i);
                      r_m(i+1)/(2*(-N/2+n-1)) * f_fourier_coeff(n,i+1)];
            C(n,i) = nonuniform_simps_rule(r_temp, f_temp);
            
            f_temp = [-r_m(i-1)/(2*n) * f_fourier_coeff(n+N/2+1,i-1); ...
                      -r_m(i)/(2*n) * (r_m(i-1)/r_m(i))^n * f_fourier_coeff(n+N/2+1,i); ...
                      -r_m(i+1)/(2*n) * (r_m(i-1)/r_m(i+1))^n * f_fourier_coeff(n+N/2+1,i+1)];
            D(n+1,i) = nonuniform_simps_rule(r_temp, f_temp);
            
            if i == 2 % Compute C^(1,2)_n and D^(M-1,M)_n using trapezoidal rule.
                C(n,1) = (delta(1))^2/(4*(-N/2 + n -1))*( f_fourier_coeff(n,2) );
                
                D(n+1,1) = -(delta(M-1))/(4*n)*( r_m(M-1) * f_fourier_coeff(n+N/2+1,M-1) + ...
                    r_m(M)*(r_m(M-1)/r_m(M))^n * f_fourier_coeff(n+N/2+1,M) );
            end
        end
        f_temp = [r_m(i-1) * f_fourier_coeff(N/2+1,i-1); ...
                  r_m(i) * f_fourier_coeff(N/2+1,i); ...
                  r_m(i+1) * f_fourier_coeff(N/2+1,i+1)];
        C(N/2+1,i) = nonuniform_simps_rule(r_temp, f_temp);
        
        % We must compute D(1,2) separately, since there is a computation of
        % 0*log(0) in the algorithm.
        if i ~= 2
            f_temp = [r_m(i-1)*log(r_m(i-1)) * f_fourier_coeff(N/2+1,i-1); ...
                      r_m(i)*log(r_m(i)) * f_fourier_coeff(N/2+1,i); ...
                      r_m(i+1)*log(r_m(i+1)) * f_fourier_coeff(N/2+1,i+1)];
            D(1,i) = nonuniform_simps_rule(r_temp, f_temp);
        end
    end
    % We must compute several more integrals separately since they are a 
    % bit different than those in the 'for loop' above.
    C(N/2+1,1) = (r_m(2))^2/2 * (f_fourier_coeff(N/2+1,2)); % Trapezoidal rule.
    r_temp = [r_m(1); r_m(2); r_m(3)];
    f_temp = [0; r_m(2)*log(r_m(2)) * f_fourier_coeff(N/2+1,2);
                 r_m(3)*log(r_m(3)) * f_fourier_coeff(N/2+1,3)];
    D(1,2) = nonuniform_simps_rule(r_temp, f_temp);
    
    D(1,1) = (delta(M-1))/2 * ( r_m(M-1)*log(r_m(M-1)) * f_fourier_coeff(N/2+1,M-1) + ...
             r_m(M)*log(r_m(M)) * f_fourier_coeff(N/2+1,M) ); % Trapezoidal rule.
end


end

