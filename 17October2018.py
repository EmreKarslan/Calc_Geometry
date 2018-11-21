
# coding: utf-8

# In[2]:

from mpl_toolkits import mplot3d


# In[21]:

x=[4,0]
y=[0,3]


# In[22]:

plt.plot(x,y)


# In[24]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt


# In[112]:

fig = plt.figure()
ax = plt.axes(projection='3d')


# In[37]:

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


# In[51]:

def my_draw_a_vector_from_origin(v):
    x=[0,v[0]]
    y=[0,v[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_origin([15,32])


# In[52]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[10,12])


# In[53]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[20,7])


# In[54]:

def my_scaler_product(a,b):
    return a[0]*b[0]+a[1]*b[1]
v=[0,4]
w=[4,0]
my_scaler_product(v,w)
my_draw_a_vector_from_origin(v)
my_draw_a_vector_from_origin(w)


# In[92]:




# In[50]:

my_draw_a_vector_from_v_to_w([5,5],[10,12])
my_draw_a_vector_from_origin([-7,5])


# In[44]:

def distance(v,w):
    return ((v[0]+w[0])**2+(v[1]+w[1])**2)**0.5
def my_vector_add(v,w):
    return [v[0]+w[0],v[1]+w[1]]
def my_vector_multiply_with_scalar(v,k):
    return [v[0]*k,v[1]*k]
def my_vector_substract(v,w):
    return [v[0]-w[0],v[1]-w[1]]


# In[46]:


a=[3,0]
b=[0,4]
c=6


# In[ ]:




# In[47]:

print("Toplam :",my_vector_add(a,b))
print("Fark :",my_vector_substract(a,b))
print("Carpim :",my_vector_multiply_with_scalar(a,c))
print("Aralarindaki Mesafe :",distance(a,b))


# In[91]:

def draw_my_line(normal_vector,point_on_line):
    a=normal_vector[0]
    b=normal_vector[1]
    c=normal_vector[2]
    x_0=point_on_line[0]
    y_0=point_on_line[1]
    z_0=point_on_line[2]
    x,y,z=[],[],[]
    for i in range(-100,100):
        x.append(x_0+i*a)       
        y.append(y_0+i*b)
        z.append(z_0+i*c)
    return[x,y,z]
def my_scaler_product(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


# In[150]:

def point_on_line(normal_vector,point_on_line,other_point):
    a_x=other_point[0]-point_on_line[0]
    a_y=other_point[1]-point_on_line[1]
    a_z=other_point[2]-point_on_line[2]
    b=normal_vector
    a=[a_x,a_y,a_z]
    
    c=my_scaler_product(a,b)/my_scaler_product(a,a)
    
    b_x=c*b[0]
    b_y=c*b[1]
    b_z=c*b[2]
    
    nearest_point_on_line=[b_x,b_y,b_z]
    return nearest_point_on_line
    


# In[164]:

p=[0,0,0]
n=[1,1,1]
other_point=[1,1,5]
n_p=point_on_line(n,p,other_point)
points=draw_my_line(n,p)

ax = plt.axes(projection='3d')
my_scaler_product(n,p)
ax.plot3D(points[0],points[1],points[2], 'red')
ax.scatter(other_point[0],other_point[1],other_point[2])
ax.scatter(n_p[0],n_p[1],n_p[2])
plt.ylim((0,25))
plt.xlim((0,25))
ax.set_zlim(-10,10)
plt.show()


# In[141]:




# In[135]:




# In[165]:

n_p


# In[ ]:



