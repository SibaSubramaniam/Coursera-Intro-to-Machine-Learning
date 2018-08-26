function [J]=backprop(X,Theta1,Theta2,m,y)
  
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
  for i=1:3
    
    x=[1 X(i,:)];
    z2=x*Theta1';
    a2=sigmoid(z2);
   
    a2=[1 a2];
    z3=a2*Theta2';
    a3=sigmoid(z3);
   
    y_vec=([1:10]==y(i))
    
    delta_3=a3-y_vec;
    delta_2=(delta_3*Theta2).*sigmoidGradient([1 z2]);
    delta_2=delta_2(2:end);
    
    Theta2_grad=Theta2_grad+delta_3'*a2;
    Theta1_grad=Theta1_grad+delta_2'*x;
    
 endfor
 size(Theta2_grad)
 size(Theta1_grad)
    
    