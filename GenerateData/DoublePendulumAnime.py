# 这段代码用于设置初值，并调用integrate求解微分方程组
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

G = 9.8
L1,L2 = 1.0, 1.0
G_L1 = G/L1
lam = L2/L1   #杆长度比L2/L1
mu = 1.0      #质量比M2/M1
M = 1+mu

# 生成时间
dt = 0.01
t = np.arange(0, 20, dt)

th1,th2 = 120.0, -10.0  #初始角度
om1,om2 = 0.0, 0.00       #初始角速度
state = np.radians([th1, om1, th2, om2])

# 其中，lam,mu,G_L1,M为全局变量
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


# 微分方程组数值解
y = integrate.odeint(derivs, state, t)

# 真实坐标
x1 = L1*np.sin(y[:, 0])
y1 = -L1*np.cos(y[:, 0])
x2 = L2*np.sin(y[:, 2]) + x1
y2 = -L2*np.cos(y[:, 2]) + y1


import matplotlib.animation as animation
# 下面为绘图过程
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# 初始化图形
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(y)),   
        interval=dt*1000, blit=True, init_func=init)

plt.show()

