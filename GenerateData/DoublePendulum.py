import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate as integrate

G = 9.8
L1,L2 = 1.0, 1.0
G_L1 = G/L1
lam = L2/L1   #杆长度比L2/L1
mu = 1.0      #质量比M2/M1
M = 1+mu

N = 1000

# 生成时间
t_max =20 
dt = t_max/N
t = np.arange(0, t_max, dt)

def derivs(state, t):
    dydx = np.zeros_like(state)
    th1,om1,th2,om2 = state
    dydx[0] = state[1]
    delta = state[2] - state[0]
    cDelta, sDelta = np.cos(delta), np.sin(delta)
    sTh1,_,sTh2,_ = np.sin(state)
    den1 = M - mu*cDelta**2
    dydx[1] = (mu * om1**2 * sDelta * cDelta
                + mu * G_L1 * sTh2 * cDelta
                + mu * lam * om2**2 * sDelta
                - M * G_L1 * sTh1)/ den1
    dydx[2] = state[3]
    den2 = lam * den1
    dydx[3] = (- mu * lam * om2**2 * sDelta * cDelta
                + M * G_L1 * sTh1 * cDelta
                - M * om1**2 * sDelta
                - M * G_L1 * sTh2)/ den2
    
    return dydx


def  GenerateDoublePendulum():
    '''生成双摆的运动轨迹，在笛卡尔坐标系中'''



    th1,th2 = 120.0, -10.0  #初始角度
    om1,om2 = 0.0, 0.00       #初始角速度
    state = np.radians([th1, om1, th2, om2])

    y = integrate.odeint(derivs, state, t)#0,1,2,3分别对应（theta1,omega1,theta2,omega2)

    # 真实坐标
    x1 = L1*np.sin(y[:, 0])
    y1 = -L1*np.cos(y[:, 0])
    
    x2 = L2*np.sin(y[:, 2]) + x1
    y2 = -L2*np.cos(y[:, 2]) + y1

    return x1,x2,y1,y2
    
if __name__ == "__main__":

    x1,x2,y1,y2 = GenerateDoublePendulum()
    px1,px2,py1,py2 = np.zeros_like(x1),np.zeros_like(x1),np.zeros_like(x1),np.zeros_like(x1)
    
    dp = 1/dt
    
    for i in range(1,x1.shape):
        px1 = (x1[i]-x1[i-1])*dp
        px2 = (x2[i]-x2[i-1])*dp
        py1 = (y1[i]-y1[i-1])*dp
        py2 = (y2[i]-y2[i-1])*dp
    
    plt.scatter(x1,y1,marker='.', label='m1')
    plt.scatter(x2,y2,marker='.', label='m2')

    plt.legend()
    plt.show()
