import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dynamic_system

def plot_trajectory(dt,X,angle):
    T = [dt*i for i in range(len(X))]
    plt.figure(1)
    plt.plot(T, X)
    plt.xlabel('time(s)')
    plt.ylabel('displacement(m)')
    plt.title('x: the displacement of the cart changes over time')
    plt.savefig('trajectory')
    plt.figure(2)
    plt.plot(T, angle)
    plt.xlabel('time(s)')
    plt.ylabel('angle(rad)')
    plt.title('\u03B8: the angle between the normal and the stick changes over time')
    plt.savefig('angle')

def animation_trajectory(dt,X,Angle):
    def update_points(num):
        x = X[num]
        angle = Angle[num]
        point_ani.set_data([x,x+math.sin(angle)*0.3],[0,math.cos(angle)*0.3])
        return point_ani,
    T = [dt*i for i in range(len(X))]
    fig = plt.figure(tight_layout=True)
    plt.xlim(-2,1)
    plt.ylim(-1,1)
    plt.xticks(np.arange(-2,2,0.25))
    plt.yticks(np.arange(-1,1,0.25))
    point_ani, = plt.plot([],[], "ro", linestyle='-')
    plt.grid(ls="--")
    ani = animation.FuncAnimation(fig, update_points, np.arange(0, len(X)), interval=0, blit=True)
    plt.show()
    # ani.save('animation.gif', writer='Pillow', fps=10000000)

def plot_states(x,angle):
    fig = plt.figure(tight_layout=True)
    plt.xlim(x-0.5,x+0.5)
    plt.ylim(-0.25,0.75)
    plt.xticks(np.arange(x-0.5,x+0.5,0.25))
    plt.yticks(np.arange(-0.25,0.75,0.25))
    plt.xlabel('X:Coordinate(m)')
    plt.ylabel('Y:Coordinate(m)')
    plt.title('Initial States')
    point_ani, = plt.plot([],[], "ro", linestyle='-')
    plt.grid(ls="--")
    point_ani.set_data([x,x+math.sin(angle)*0.3],[0,math.cos(angle)*0.3])
    plt.text(x, -0.05, 'Cart', ha='center', va='bottom', fontsize=10.5)
    plt.text(x+math.sin(angle)*0.3, math.cos(angle)*0.3+0.01, 'Pendulum', ha='center', va='bottom', fontsize=10.5)
    plt.annotate('v = 2',xy=(x+0.2,0),xytext=(x-0.08,0),xycoords='data',arrowprops=dict(facecolor='black', width=0.1, headwidth=5, headlength=5,shrink=1))
    plt.savefig('init')

if  __name__ == "__main__":
    init_value = [-2,2,1,0]
    plot_states(init_value[0],init_value[2])

    dt, t = 0.01, 10
    new_system = dynamic_system.DynamicSystem(init_value, dt, t)
    K = new_system.LQR_control()
    # with control law
    # x1,x2,x3,x4 = new_system.simulation_with_feedback(K)
    # without control law
    # x1,x2,x3,x4 = new_system.simulation_middle_point()
    # plot_trajectory(dt,x1,x3)
    # animation_trajectory(dt,x1,x3)