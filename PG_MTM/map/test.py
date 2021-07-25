
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from ctypes.util import find_library
find_library('geos_c')
import seaborn

fig = plt.figure()
ax = fig.add_subplot(111)


p1 = Polygon([(-1.1,2.98), (2.32,3.66), (2.98,1.58), (-0.74,0.82)])
x1,y1 = p1.exterior.xy
ax.plot(x1, y1, color='black')

p2 = Polygon([(4.82,0.02), (6.42,1.74), (2.32,3.66), (2.98,1.58)])
x2, y2 = p2.exterior.xy
ax.plot(x2, y2, color='black')


p3 = Polygon([(6.22,-0.92), (8.42,-0.34), (6.42,1.74), (4.82,0.02)])
x3,y3 = p3.exterior.xy
ax.plot(x3, y3, color='black')




plt.show()
exit()



xAxes = [0, 100, 10]
yAxes = [0, 100, 10]

p2 = Polygon([(10, 20), (15, 30), (25, 5), (17 ,3)])
x2,y2 = p2.exterior.xy

p3 = Polygon([(40, 40), (50, 50), (40, 50)])
x3,y3 = p3.exterior.xy



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(xAxes[0], xAxes[1])
ax.set_ylim(yAxes[0], yAxes[1])
ax.set_xticks([i for i in range(xAxes[0], xAxes[1], xAxes[2])])
ax.set_yticks([i for i in range(yAxes[0], yAxes[1], yAxes[2])])
ax.plot(x2, y2)
ax.plot(x3, y3)
ax.set_title('Polygon')

ax.scatter([1], [2], color='black')

plt.show()