def get_input(prompt):
  return list(map(int, input(prompt).split()))

def init_weights():
  return 0, 0 ,0 

def update_weights(A,B,Y, weight1, weight2, bias):
  for i in range (len(Y)):
    delta_weight1 = A[i]*Y[i]
    delta_weight2 = B[i]*Y[i]
    delta_bias = Y[i]

    weight1 = weight1 + delta_weight1
    weight2 = weight2 + delta_weight2
    bias = bias + delta_bias
  return weight1, weight2, bias

def display_weights(weight1, weight2, bias):
  print(f'The weight 1 is {weight1}, the weight 2 is {weight2} and the bias is {bias}!')

def hebb_rule():
  A = get_input("Enter the A values of the truth table, separated by spaces: ")
  B = get_input("Enter the B values of the truth table, separated by spaces: ")
  Y = get_input("Enter the Y values of the truth table, separated by spaces: ")
  
  weight1, weight2, bias = init_weights()
  weight1, weight2, bias = update_weights(A, B, Y, weight1, weight2, bias)
  display_weights(weight1, weight2, bias)
    
hebb_rule()