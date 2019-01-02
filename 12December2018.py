#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
a=5
b=3
x=np.linspace(-5.0,5.0,num=50)


# In[3]:


s=range(-3,3,1)


# In[4]:


y=(b**2*(1-(x**2)/(a**2)))**.5


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.plot(x,y)
plt.plot(x,-y)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')


fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = X**2 + Y**2
Z = np.sin(R)
z2=-5+2*X+4*Y
plt.xlim(-5,5)  
plt.ylim(-5,5) 

ax.plot_surface(X, Y, R, rstride=1,cstride=1, cmap='hot')
ax.set_zlim(0,30)
ax.plot_surface(X, Y, z2, rstride=1, cstride=1, color='b')

plt.show()


# In[97]:






# In[6]:


import math

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = X**2 + Y**2
Z = np.sin(R)
z2=-5+2*X+4*Y
plt.xlim(-5,5)  
plt.ylim(-5,5) 

ax.plot_surface(X, Y, R, rstride=1,cstride=1, cmap='hot')
ax.set_zlim(0,30)
ax.plot_surface(X, Y, z2, rstride=1, cstride=1, color='b')

plt.show()
math.exp(1)


# In[ ]:




