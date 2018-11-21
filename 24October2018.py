
# coding: utf-8

# In[56]:

from mpl_toolkits import mplot3d


# In[57]:

x=[4,0]
y=[0,3]


# In[58]:

plt.plot(x,y)


# In[59]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt


# In[60]:

fig = plt.figure()
ax = plt.axes(projection='3d')


# In[97]:

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


# In[45]:

def my_draw_a_vector_from_origin(v):
    x=[0,v[0]]
    y=[0,v[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_origin([15,32])


# In[46]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.xlim(-5,15)
    plt.ylim(-5,15)
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[10,12])


# In[47]:

def my_draw_a_vector_from_v_to_w(v,w):
    x=[v[0],w[0]]
    y=[v[1],w[1]]
    plt.plot(x,y)
my_draw_a_vector_from_v_to_w([5,7],[20,7])


# In[48]:

def my_scaler_product(a,b):
    return a[0]*b[0]+a[1]*b[1]
v=[0,4]
w=[4,0]
my_scaler_product(v,w)
my_draw_a_vector_from_origin(v)
my_draw_a_vector_from_origin(w)


# In[92]:




# In[49]:

my_draw_a_vector_from_v_to_w([5,5],[10,12])
my_draw_a_vector_from_origin([-7,5])


# In[50]:

def distance(v,w):
    return ((v[0]+w[0])**2+(v[1]+w[1])**2)**0.5
def my_vector_add(v,w):
    return [v[0]+w[0],v[1]+w[1]]
def my_vector_multiply_with_scalar(v,k):
    return [v[0]*k,v[1]*k]
def my_vector_substract(v,w):
    return [v[0]-w[0],v[1]-w[1]]


# In[51]:


a=[3,0]
b=[0,4]
c=6


# In[ ]:




# In[52]:

print("Toplam :",my_vector_add(a,b))
print("Fark :",my_vector_substract(a,b))
print("Carpim :",my_vector_multiply_with_scalar(a,c))
print("Aralarindaki Mesafe :",distance(a,b))


# In[53]:

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


# In[88]:

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
    


# In[96]:

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


# In[110]:

def my_scaler_product(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def my_length_function(a):
    return my_scaler_product(a,a)**.5
def three_dimensional_point(point):
    # Data for three-dimensional scattered points
    ax.scatter3D(point[0], point[1], point[2], c=zdata, cmap='gray');

def plane_point(plane,point):
    
    #draw plane
    #draw two point
    #prind distance(d)
    plane_normal=plane[0],plane[1],plane[2]
    d=my_scaler_product(plane_normal,point) / my_length_function(plane)
    t=(-plane[3]-my_scaler_product(plane_normal,point))/my_scaler_product(plane_normal,plane_normal)
    p_0=[point[0]+t*plane[0],point[1]+t*plane[1],point[2]+t*plane[2]]
    # Data for a three-dimensional line
    zline = [p_0[0],0,0]
    xline = [p_0[1],0,0]
    yline = [p_0[2],0,0]
    ax.plot3D(xline, yline, zline, 'blue')
    three_dimensional_point(point)
    yy, zz = np.meshgrid(range(2), range(2))
    xx = yy*0

    ax = plt.subplot(projection='3d')
    ax.plot_surface(xx, yy, zz)
    plt.show()
    return d,t,p_0
    #t=0
    #p_0=0
    #plane,point,p_0


# In[109]:

fig = plt.figure()
ax = plt.axes(projection='3d')


# In[111]:

plane_1=[1,2,3,-6]
point_1=[4,2,10]
plane_point(plane_1,point_1)


# In[99]:


point1  = np.array([0,0,0])
normal1 = np.array([1,-2,1])

point2  = np.array([0,-4,0])
normal2 = np.array([0,2,-8])

point3  = np.array([0,0,1])
normal3 = np.array([-4,5,9])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d1 = -np.sum(point1*normal1)# dot product


# create x,y
xx, yy = np.meshgrid(range(30), range(30))

# calculate corresponding z
z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx,yy,z1, color='blue')

plt.show()


# In[108]:

plt.plot(x,y,2)


# In[ ]:



