
# coding: utf-8

# In[18]:

from mpl_toolkits.mplot3d import Axes3D


# In[8]:

x=[4,0]
y=[0,3]
get_ipython().magic('matplotlib inline')


# In[10]:

plt.plot(x,y)


# In[12]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# In[19]:

fig = plt.figure()
ax = plt.axes(projection='3d')


# In[20]:

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = [3,0,0]
xline = [4,0,0]
yline = [5,0,0]
ax.plot3D(xline, yline, zline, 'blue')

# Data for three-dimensional scattered points
zdata = 1
xdata = 1
ydata = 1
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='gray');


# In[91]:

def my_draw_a_vector_from_origin(v):
    x=[0,v[0]]
    y=[0,v[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_origin([15,32])


# In[95]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[10,12])


# In[87]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[20,7])


# In[93]:

def my_scaler_product(a,b):
    return a[0]*b[0]+a[1]*b[1]
v=[0,4]
w=[4,0]
my_scaler_product(v,w)
my_draw_z_vector_from_origin(v)
my_draw_a_vector_from_origin(w)


# In[92]:




# In[108]:

my_draw_a_vector_from_v_to_w([5,5],[10,12])
my_draw_a_vector_from_origin([-7,5])


# In[124]:

def distance(v,w):
    return ((v[0]+w[0])**2+(v[1]+w[1])**2)**0.5
def my_vector_add(v,w):
    return [v[0]+w[0],v[1]+w[1]]
def my_vector_multiply_with_scalar(v,k):
    return [v[0]*k,v[1]*k]
def my_vector_substract(v,w):
    return [v[0]-w[0],v[1]-w[1]]


# In[126]:


a=[3,0]
b=[0,4]
c=6


# In[ ]:




# In[127]:

print("Toplam :",my_vector_add(a,b))
print("Fark :",my_vector_substract(a,b))
print("Carpim :",my_vector_multiply_with_scalar(a,c))
print("Aralarindaki Mesafe :",distance(a,b))


# In[119]:

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0,30, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray' , lw=3)




# In[52]:

point_test= np.array([1,-1,1])
normal_test = np.array([1,-2,1])

sum(point_test*normal_test)


# In[84]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

point1 = np.array([0,0,0])
normal1 = np.array([1,-2,1])

point2 = np.array([0,-4,0])
normal2 = np.array([0,2,-8])

point3 = np.array([0,0,1])
normal3 = np.array([-4,5,9])

d1 = -np.sum(point1*normal1) #dot product
xx, yy = np.meshgrid(range(5),range(5))
z1 = (-normal1[0]*xx-normal1[1]*yy-d1)/normal1[2]
z2 = (-normal2[0]*xx-normal2[1]*yy-d1)/normal2[2]
z3 = (-normal3[0]*xx-normal3[1]*yy-d1)/normal3[2]


get_ipython().magic('matplotlib inline')
plt3d=plt.figure().gca(projection='3d')

plt3d.plot_surface(xx,yy,z1, color='blue')
plt3d.plot_surface(xx,yy,z2, color='green')
plt3d.plot_surface(xx,yy,z3, color='red')


# In[60]:

d1 = -np.sum(point1*normal1) #dot product
d2 = -np.sum(point2*normal2) #dot product
d3 = -np.sum(point3*normal3) #dot product
d1 , d2 , d3


# In[61]:

# 4x+5y+6z+1=0
# -<4,5,6>.<x,y,z>=d


# In[72]:

xx, yy = np.meshgrid(range(5),range(5))


# In[75]:

z1 = (-normal1[0]*xx-normal1[1]*yy-d1)/normal1[2]


# In[120]:

fig= plt.figure()
ax=fig.add_subplot(111, projection='3d')

n = 1000
theta_max = 8 * np.pi
theta = np.linspace(0,theta_max,n)
x= np.sin(theta)
y= np.cos(theta)
z= theta

ax.plot(x,y,z,'b',lw=2)


# In[ ]:



