from matplotlib import pyplot as plt
import numpy as np
import math

class LQR:
    def __init__(self, A, B, Q, R, dt, t):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.dt = dt
        self.t = t
    
    def df(self, P):
        temp = self.B.dot(np.linalg.inv(self.R).dot(self.B.T))
        P_ = -P.dot(self.A)-self.A.T.dot(P) + P.dot(temp.dot(P))-self.Q
        return P_
    
    def get_k(self):
        P = self.Q
        K_list = []
        for i in range(int(self.t/self.dt)):
            P_ = self.df(P)
            P = P - self.dt*self.df(P - self.dt / 2 * P_)
            K_list.append(-np.linalg.inv(self.R).dot(self.B.T.dot(P)))
        K_list.reverse()
        return K_list

class DynamicSystem:
    def __init__(self, init_value, dt, t):
        self.init_value = init_value
        self.dt = dt
        self.t = t
        self.M = 0.5
        self.m = 0.2
        self.L = 0.3
        self.I = 0.006
        self.b = 0
        self.c = 0
        self.g = 10.0
    
    def df(self, x1, x2, x3, x4, F = 0):
        M,m,L,I,b,c,g = self.M, self.m, self.L, self.I, self.b, self.c, self.g
        temp_ = (I+m*L*L)*(M+m) - m*m*L*L*math.cos(x3)*math.cos(x3)
        
        temp_1 = [m*L*c, (I+m*L*L)*m*L, -(I+m*L*L)*b, -m*m*L*L*g, (I+m*L*L)]
        temp_2 = [(M+m)*m*g*L, m*L*b, -m*m*L*L, -(M+m)*c, -m*L]
        x1_ = x2
        x2_ = (temp_1[0]*math.cos(x3)*x4 + temp_1[1]*math.sin(x3)*x4*x4 + temp_1[2]*x2 + temp_1[3]*math.cos(x3)*math.sin(x3))/temp_ + temp_1[4]/temp_*F
        x3_ = x4
        x4_ = (temp_2[0]*math.sin(x3) + temp_2[1]*math.cos(x3)*x2 + temp_2[2]*math.sin(x3)*math.cos(x3)*x4*x4 +temp_2[3]*x4)/temp_ + temp_2[4]/temp_*math.cos(x3)*F
        return (x1_, x2_, x3_, x4_)

    def df_linear(self, x1, x2, x3, x4, F = 0):
        M,m,L,I,b,c,g = self.M, self.m, self.L, self.I, self.b, self.c, self.g
        temp_ = (I+m*L*L)*(M+m) - m*m*L*L
        temp_1 = [m*L*c, -(I+m*L*L)*b, -m*m*L*L*g, (I+m*L*L)]
        temp_2 = [(M+m)*m*g*L, m*L*b, -(M+m)*c, -m*L]
        x1_ = x2
        x2_ = (temp_1[0]*x4 + temp_1[1]*x2 + temp_1[2]*x3)/temp_ + temp_1[3]/temp_*F
        x3_ = x4
        x4_ = (temp_2[0]*x3 + temp_2[1]*x2 + temp_2[2]*x4)/temp_ + temp_2[4]/temp_*F
        return (x1_, x2_, x3_, x4_)

    def simulation_forward_Eular(self):
        x1,x2,x3,x4 = self.init_value[0], self.init_value[1], self.init_value[2], self.init_value[3]
        dt = self.dt
        t = self.t
        x1_list,x2_list,x3_list,x4_list = [], [], [], []
        for i in range(int(t/dt)):
            x1_, x2_, x3_, x4_ = self.df(x1,x2,x3,x4)
            x1 += dt * x1_
            x2 += dt * x2_
            x3 += dt * x3_
            x4 += dt * x4_
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            x4_list.append(x4)
        return (x1_list, x2_list, x3_list, x4_list)

    def simulation_middle_point(self):
        x1,x2,x3,x4 = self.init_value[0], self.init_value[1], self.init_value[2], self.init_value[3]
        dt = self.dt
        t = self.t
        x1_list,x2_list,x3_list,x4_list = [], [], [], []
        for i in range(int(t/dt)):
            x1_, x2_, x3_, x4_ = self.df(x1,x2,x3,x4)
            x1_, x2_, x3_, x4_ = self.df(x1 + 0.5*dt*x1_, x2 + 0.5*dt*x2_, x3 + 0.5*dt*x3_, x4 + 0.5*dt*x4_)
            x1 += dt * x1_
            x2 += dt * x2_
            x3 += dt * x3_
            x4 += dt * x4_
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            x4_list.append(x4)
        return (x1_list, x2_list, x3_list, x4_list)

    def simulation_with_feedback(self, K):
        x1,x2,x3,x4 = self.init_value[0], self.init_value[1], self.init_value[2], self.init_value[3]
        dt = self.dt
        t = self.t
        x1_list,x2_list,x3_list,x4_list = [], [], [], []
        for i in range(int(t/dt)):
            F = K[i][0][0]*x1 + K[i][0][1]*x2 + K[i][0][2]*x3 + K[i][0][3]*x4
            x1_, x2_, x3_, x4_ = self.df(x1,x2,x3,x4,F)
            F = K[i][0][0]*(x1 + 0.5*dt*x1_) + K[i][0][1]*(x2 + 0.5*dt*x2_) + K[i][0][2]*(x3 + 0.5*dt*x3_) + K[i][0][3]*(x4 + 0.5*dt*x4_)
            x1_, x2_, x3_, x4_ = self.df(x1 + 0.5*dt*x1_, x2 + 0.5*dt*x2_, x3 + 0.5*dt*x3_, x4 + 0.5*dt*x4_, F)
            x1 += dt * x1_
            x2 += dt * x2_
            x3 += dt * x3_
            x4 += dt * x4_
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            x4_list.append(x4)
        return (x1_list, x2_list, x3_list, x4_list)

    def LQR_control(self):
        M,m,L,I,b,c,g = self.M, self.m, self.L, self.I, self.b, self.c, self.g
        temp_ = (I+m*L*L)*(M+m) - m*m*L*L
        A = np.zeros([4,4], dtype = float)
        A[0,1] = 1
        A[1,1] = -(I+m*L*L)*b / temp_
        A[1,2] = -m*m*L*L*g / temp_
        A[1,3] = m*L*c / temp_
        A[2,3] = 1
        A[3,1] = m*L*b / temp_
        A[3,2] = (M+m)*m*g*L / temp_
        A[3,3] = -(M+m)*c / temp_

        B = np.zeros([4,1], dtype = float)
        B[1,0] = I+m*L*L / temp_
        B[3,0] = -m*L / temp_

        Q = np.eye(4, dtype = float)
        Q[0,0] = 1
        Q[1,1] = 1
        Q[2,2] = 1
        Q[3,3] = 1

        R = np.zeros([1,1], dtype = float)
        R[0,0] = 1

        LQR_solver = LQR(A, B, Q, R, self.dt, self.t)
        return LQR_solver.get_k()