import numpy as np
import matplotlib.pyplot as plt




# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(m, b, points):
     Total_Error = 0
     for i in range(0, len(points)):
          x = points[i, 0]
          y = points[i, 1]
          Total_Error += (y - (m * x + b)) ** 2
     return Total_Error / float(len(points))

def Step_Gradient(m_current, b_current, points, learning_rate):
     m_gradient = 0
     b_gradient = 0
     N = float(len(points))
     for i in range (len(points)):
          x = points[i,0]
          y = points[i,1] 
          m_gradient += -2/N * x * ( y - (m_current * x + b_current))
          b_gradient += -2/N * (y - ((m_current * x) + b_current))
     new_m = m_current - (learning_rate * m_gradient)
     new_b = b_current - (learning_rate * b_gradient)
     return[new_m,new_b]

def Gradient_run(m_starting, b_starting, points, learning_rate, interations):
     m = m_starting
     b = b_starting
     for i in range (0, interations):
          [m, b] = Step_Gradient(m, b, points, learning_rate)
     return m, b          
     

def run():
     initial_b = 0
     initial_m = 0
     learning_rate = 0.0001
     num_iterations = 1000
     points = np.genfromtxt("data.csv",delimiter=",")
     #print(points)

     print("Starting Gradient descent at: m = {0}, b = {1}, Total_Error = {2}".format(initial_m,initial_b, compute_error_for_line_given_points(initial_m,initial_b, points)))
     print("Strarting ...")

     [m, b] = Gradient_run(initial_m, initial_b, points, learning_rate, num_iterations)
     print("Gradient descent after learning: m = {0}, b = {1}, Total_Error = {2}".format(m,b, compute_error_for_line_given_points(m,b, points)))
     plt.plot(points)


if __name__ == '__main__':
     run()
     print("{:*^40s}".format("Edit by Hoang Huynh"))
