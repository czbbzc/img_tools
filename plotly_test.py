import plotly.graph_objects as go
import numpy as np
# np.random.seed(1)
 
N = 70
 
# fig = go.Figure(data=[go.Mesh3d(x=(70*np.random.randn(N)),
#                    y=(55*np.random.randn(N)),
#                    z=(40*np.random.randn(N)),
#                    opacity=0.5, # 设置透明度
#                    color='rgba(244,22,100,0.6)'
#                   )])

fig = go.Figure(data=[go.Scatter3d(x=(70*np.random.randn(N)),
                   y=(55*np.random.randn(N)),
                   z=(40*np.random.randn(N)),
                   mode='text',
                   opacity=0.5, # 设置透明度
                   marker=dict(color='blue',size=3)
                #    color='rgba(244,22,100,0.6)'
                  )])

            # type="scatter3d",
            # x=[float(n) for n in center[:,0]],
            # y=[float(n) for n in center[:,1]],
            # z=[float(n) for n in center[:,2]],
            # mode="markers",
            # marker=dict(color=color,size=3),

# fig.update_layout(scene_aspectmode='cube')

# 坐标轴设置
fig.update_layout(
    scene = dict(
                    xaxis = dict(nticks=4, range=[-100,100],),
                    yaxis = dict(nticks=4, range=[-100,100],), # 标记4个标签值
                    zaxis = dict(nticks=4, range=[-100,100],),),
    width=1000,
    scene_aspectmode='cube',
    margin=dict(r=10, l=10, b=10, t=10))
 
fig.show()