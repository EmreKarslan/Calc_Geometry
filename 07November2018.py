
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

point1= np.array([0,0,0])
normal1= np.array([1,-2,1])

d1=-np.sum(point1*normal1)

xx,yy=np.meshgrid(range(5),range(5))

z1=(-normal1[0]*xx-normal1[1]*yy -d1)*1./normal1[2]
get_ipython().magic('matplotlib inline')
plt3d=plt.figure().gca(projection='3d')
plt3d.plot_surface(xx,yy,z1, color='green')



# In[14]:

point1= np.array([4,5,6])
normal1= np.array([4,5,6])

d1=-np.sum(point1*normal1)

xx,yy=np.meshgrid(range(5),range(5))

z1=(-normal1[0]*xx-normal1[1]*yy -d1)*1./normal1[2]
get_ipython().magic('matplotlib inline')
plt3d=plt.figure().gca(projection='3d')
plt3d.plot_surface(xx,yy,z1, color='green')



# In[8]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
point1= np.array([7,8,9])
normal1= np.array([7,8,9])

d1=-np.sum(point1*normal1)

xx,yy=np.meshgrid(range(5),range(5))

z1=(-normal1[0]*xx-normal1[1]*yy -d1)*1./normal1[2]
get_ipython().magic('matplotlib notebook')
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.gca(projection='3d').plot_surface(xx,yy,z1, color='green')


# In[40]:

def my_F(my_Matris):
    
    get_ipython().magic('matplotlib notebook')
    
    point1= np.array([0,0,0])
    normal1= np.array(my_Matris[0])
    normal2= np.array(my_Matris[1])
    normal3= np.array(my_Matris[2])
    
    d1=-np.sum(point1*normal1)
    d2=-np.sum(point1*normal2)
    d3=-np.sum(point1*normal3)

    xx,yy=np.meshgrid(range(5),range(5))
    z1=(-normal1[0]*xx-normal1[1]*yy -d1)*1./normal1[2]
    z2=(-normal2[0]*xx-normal2[1]*yy -d2)*1./normal2[2]
    z3=(-normal3[0]*xx-normal3[1]*yy -d3)*1./normal3[2]
    plt3d=plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx,yy,z1, color='blue')
    plt3d.plot_surface(xx,yy,z2, color='green')
    plt3d.plot_surface(xx,yy,z3, color='red')


# In[41]:

my_M=np.zeros((3,3))
my_M[0,0]=4
my_M[0,1]=5
my_M[0,2]=6
my_M[1,0]=7
my_M[1,1]=8
my_M[1,2]=9
my_M[2,0]=10
my_M[2,1]=11
my_M[2,2]=12

my_F(my_M)


# In[37]:

my_M[0]=my_M[0]-my_M[2]*my_M[1]
my_F(my_M)


# In[38]:

my_M[1]=my_M[1]-my_M[0]*my_M[2]
my_F(my_M)


# In[39]:

my_M[2]=my_M[2]-my_M[0]*my_M[1]
my_F(my_M)


# In[ ]:



