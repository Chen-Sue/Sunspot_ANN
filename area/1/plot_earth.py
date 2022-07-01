
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 6))
r = 90
x,y = (0,0)
a_x = np.arange(0,2*np.pi,0.01)
a = x+r*np.cos(a_x)
b = y+r*np.sin(a_x)
plt.plot(a,b,color='black',linestyle='-')
plt.plot(a,-b,color='black',linestyle='-')
import config
z = config.lat_sep[1:].reshape(-1,1)
for i in np.arange(len(z)):
    plt.axhline(y=z[i], color='r', linestyle='-')
plt.yticks([-120,-90,-60,-30,0,30,60,90],
    ['120S','90S','60S','30S','EQ','30N','60N','90N'], size=14)
plt.show()

# ax = Axes3D(fig)
# # u = np.linspace(0, 2 * np.pi, 100)
# # v = np.linspace(0, np.pi, 100)
# # u, v = np.meshgrid(u, v)
# X = np.sin(v) * np.cos(u)
# Y = np.sin(v) * np.sin(u)
# Z = np.cos(v)
# ax.plot_surface(X, Y, Z, cmap='rainbow')


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# t = np.linspace(0, np.pi * 2, 100)
# s = np.linspace(0, np.pi, 100)

# t, s = np.meshgrid(t, s)
# x = np.cos(t) * np.sin(s)
# y = np.sin(t) * np.sin(s)
# z = np.cos(s)
# ax = plt.subplot(111, projection='3d')
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
# # 去除刻度

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np 

# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # u = np.linspace(0, 2*np.pi, 100)
# # v = np.linspace(0, np.pi, 100)
# # x = 10 * np.outer(np.cos(u), np.sin(v))
# # y = 10 * np.outer(np.sin(u), np.sin(v))
# # # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
# # # x = np.linspace(-10, 10, 1000)
# # # y = np.linspace(-10, 10, 1000)
# # z = np.add(x, y)
# # ax.plot(x, y, 1)
# # plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # plt = fig.add_subplot(111, projection='3d')
# u = np.linspace(0, 2*np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
# # ax.plot_surface(x, y, z, cmap='rainbow')
# import matplotlib
# import config
# # print(x.shape, y.shape, z.shape)
# # z = config.lat_sep[1:].reshape(-1,1)
# # ax.contour(x, y, z, cmap=matplotlib.cm.Accent, linewidths=2)
# z_line = np.array(config.lat_sep[1:])
# # z_line = np.repeat(z_line, len(z_line), axis=1)
# u = np.linspace(0, 2*np.pi, len(z_line))
# v = np.linspace(0, np.pi, len(z_line))
# x_line = 10 * np.outer(np.cos(u), np.sin(v))
# y_line = 10 * np.outer(np.sin(u), np.sin(v))
# print(x_line, y_line.shape, z_line.shape)

# ax.plot3D(x_line[:, 0], y_line[:, 0], z_line, 'gray')
# ax.plot3D(x_line[:, 1], y_line[1, :], z_line, 'gray')
# # plt.plot(x, y, np.ones_like(z))

# # plt.grid(False)
# # ax.plot_surface(x, y, z, color='b')
# # plt.xaxis.set_major_locator(plt.NullLocator())
# # plt.yaxis.set_major_locator(plt.NullLocator())
# # plt.zaxis.set_major_locator(plt.NullLocator())
# plt.show()