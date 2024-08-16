import threading
import requests
import numpy as np
import sys
import json
import time
import os



def send_NBV(location,u,v):
    x = location[0]
    y = location[1]
    z = location[2]
    u = u/np.pi*180.
    v = v/np.pi*180.
    response = requests.get("http://10.25.220.109:7200/", "x="+str(x)+"&y="+str(y)+"&z="+str(z)+"&u="+str(u)+"&v="+str(v))
    return response.text


object_center_list = []
bound_box_list = []
view_num_list = []

## room 1
object_center_list.append(np.array([0,1.7,-2])) 
bound_box_list.append(np.array([[-1.1,0.6],[1.4,2.0],[-3.5,0]]))
view_num_list.append(100)
## room 2
object_center_list.append(np.array([-3.5,1.7,-2.75])) 
bound_box_list.append(np.array([[-4,-3.0],[1.4,2.0],[-3.5,-2.0]]))
view_num_list.append(50)
## room 3
object_center_list.append(np.array([0.4,1.7,2.25])) 
bound_box_list.append(np.array([[0,0.8],[1.4,2.0],[1.5,3.0]]))
view_num_list.append(30)
## room 4
object_center_list.append(np.array([-1.4,1.7,2.25])) 
bound_box_list.append(np.array([[-1.6,-1.2],[1.4,2.0],[1.5,3.0]]))
view_num_list.append(30)
## room 5
object_center_list.append(np.array([-3.75,1.7,3.4])) 
bound_box_list.append(np.array([[-4,-3.5],[1.4,2.0],[2.8,4.0]]))
view_num_list.append(30)
## room 6
object_center_list.append(np.array([-6.75,1.7,-4.1])) 
bound_box_list.append(np.array([[-7.5,-6.0],[1.4,2.0],[-5,-3.2]]))
view_num_list.append(50)
## room 7
object_center_list.append(np.array([-10,1.7,-4.1])) 
bound_box_list.append(np.array([[-10.7,-9.3],[1.4,2.0],[-5,-3.2]]))
view_num_list.append(50)
## room 8
object_center_list.append(np.array([-10.5,1.7,-1.2])) 
bound_box_list.append(np.array([[-10.6,-10.4],[1.2,2.5],[-1.3,-1.1]]))
view_num_list.append(20)
## room 9
object_center_list.append(np.array([-8,1.7,4])) 
bound_box_list.append(np.array([[-9,-7],[1.4,2.0],[2.5,5]]))
view_num_list.append(100)
## room 10
object_center_list.append(np.array([-14,1.7,-3.5])) 
bound_box_list.append(np.array([[-15,-13],[1.2,2.3],[-5,-2]]))
view_num_list.append(100)
## room 11
object_center_list.append(np.array([-12.5,1.7,3])) 
bound_box_list.append(np.array([[-12.6,-12.4],[1.2,2.3],[2,4]]))
view_num_list.append(30)
## room 12
object_center_list.append(np.array([-18,1.7,-1.0])) 
bound_box_list.append(np.array([[-19,-17],[1.2,2.3],[-2,0]]))
view_num_list.append(80)
## room 13
object_center_list.append(np.array([-16.5,1.7,3.5])) 
bound_box_list.append(np.array([[-18,-15],[1.2,2.3],[2.5,4.5]]))
view_num_list.append(100)
## room 14
object_center_list.append(np.array([-14.5,1.7,6.2])) 
bound_box_list.append(np.array([[-14.6,-14.4],[1.2,2.3],[6.1,6.3]]))
view_num_list.append(20)



for k in range(0,5,1):
    object_center = object_center_list[k]
    bound_box = bound_box_list[k]
    num = int(view_num_list[k]*1)
    for i in range(num):
        x = np.random.uniform(bound_box[0,0],bound_box[0,1])
        y = np.random.uniform(bound_box[1,0],bound_box[1,1])
        z = np.random.uniform(bound_box[2,0],bound_box[2,1])

        dx,dy,dz =  object_center[0]-x, object_center[1]-y,object_center[2]-z
        # u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
        u = np.arctan2(-dy,0.4) + np.random.randn(1)[0]*5/180*np.pi
        # v = np.arctan2(dx,dz)
        v =  (np.random.uniform(0,36,1)[0]*10)/180*np.pi

        view = [x,y,z,u,v]
        print(k,i,view)
        send_NBV(view[0:3],view[3],view[4])
        time.sleep(0.7)



# Test ###
# view = [2,1.5,-3.0,0,0]
# send_NBV(view[0:3],view[3],view[4])

# view = [-7,1.2,0,0,0.5*np.pi]
# send_NBV(view[0:3],view[3],view[4])

# view = [-0.5,1.7,-2.5,0,0.5*np.pi]
# send_NBV(view[0:3],view[3],view[4])

# view = [-0.05850094554669605, 1.6861298547774526, -1.5773021063197952, 0.10114289464621781, 3.9813864929416747]
# send_NBV(view[0:3],view[3],view[4])


# ## sample test views from scene hm3d_00804
# views_dir = "/mnt/dataset/zengjing/monosdf_planning/data/Views/childroom.txt"
# views = np.loadtxt(views_dir)
# print(views, views.shape)
# for i in range(views.shape[0]):
#     send_NBV(views[i,0:3].tolist(),views[i,3],views[i,4])
#     print(i)
#     time.sleep(0.4)
    


view = [0.0,1.0,-3.0,0,0]
send_NBV(view[0:3],view[3],view[4])