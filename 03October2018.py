
# coding: utf-8

# In[23]:

from mpl_toolkits import mplot3d


# In[10]:

x=[4,0]
y=[0,3]
get_ipython().magic('matplotlib inline')


# In[26]:

plt.plot(x,y)


# In[24]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# In[27]:

fig = plt.figure()
ax = plt.axes(projection='3d')


# In[68]:

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


# In[ ]:



