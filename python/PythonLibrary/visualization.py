'''
This is one of the basics of Tool-Collection

Selected plots and animations
'''

from typing import List
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm
from .general import Pick_Significant

# vector x, matrix y, (x[i], y[j,i]) is the coordinate of point_i in j-th series
# when there is only 1 series, y can alternatively be passed as a vector
# PlotType can be chosen from scatter, line
def Plot2D(x, y,
title=' ', xlabel='x', ylabel='y',
color='black', PlotType='line') -> None:
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if(PlotType=='scatter'):
        if len(y.shape) == 1: plt.scatter(x, y, color = color)
        else:
            if len(color) == y.shape[0]:
                for i in range(y.shape[0]): plt.scatter(x, y[i,:], color = color[i])
            else:
                for i in range(y.shape[0]): plt.scatter(x, y[i,:])
    else:
        if len(y.shape) == 1: plt.plot(x, y, color = color)
        else:
            if len(color) == y.shape[0]:
                for i in range(y.shape[0]): plt.plot(x, y[i,:], color = color[i])
            else:
                for i in range(y.shape[0]): plt.plot(x, y[i,:])
    plt.show()

# PlotType can be chosen from heatmap, scatter, line, surface
# for heatmap:
#     vector x & y, matrix z, (x[j],y[i],z[i,j]) is the coordinate of point_ij
# else:
#     vector x & y & z, (x[i],y[i],z[i]) is the coordinate of point_i
def Plot3D(x, y, z,
title=' ', xlabel='x', ylabel='y', zlabel='z',
PlotType='heatmap', color='black', colormap='',
heat_map_annotation=False, scatter_size=100) -> None:
    if PlotType == 'heatmap':
        assert x.shape[0] == z.shape[1]
        assert y.shape[0] == z.shape[0]
        fig, ax = plt.subplots()
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        im = ax.imshow(z, cmap=colormap)
        ax.set_xticks(numpy.arange(x.shape[0])); ax.set_xticklabels(x)
        ax.set_yticks(numpy.arange(y.shape[0])); ax.set_yticklabels(y)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.colorbar(im)
        if heat_map_annotation:
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    text = ax.text(j, i, z[i, j], ha="center", va="center", color="w")
        fig.tight_layout()
    else:
        ax = plt.subplot(111, projection='3d')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
        if PlotType == 'scatter': ax.scatter(x, y, z, color=color, s=scatter_size, depthshade=True)
        elif PlotType == 'line': ax.plot(x, y, z, color=color)
        else:
            if colormap == '': ax.plot_trisurf(x, y, z, color=color)
            else: ax.plot_trisurf(x, y, z, cmap=colormap)
    plt.show()

# vector t & x, x.shape[0] by level by t.shape[0] 3rd-order tensor y
# y[i,k,j] is value on k-th level at x[i] position at t[j] time
# when there is only 1 level, y can alternatively be passed as a matrix
# default FPS = 25
def Animate2D(t, x, y,
title=' ', xlabel='x', ylabel='y', color='black',
speed=1.0, save=False, FileName='2D', show=True):
    if len(y.shape) == 2:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xlim(numpy.amin(x),numpy.amax(x)); ax.set_ylim(numpy.amin(y),numpy.amax(y))
        temp, = ax.plot(x,y[:,0],color=color); line=[temp]
        def animate(i):
            line[0].set_ydata(y[:,i])
            return line
    else:
        fig, ax = plt.subplots(y.shape[1], 1, squeeze=True)
        xmin=numpy.amin(x); xmax=numpy.amax(x); ymin=numpy.amin(y); ymax=numpy.amax(y)
        line=[]
        for j in range(y.shape[1]):
            ax[j].set_title(title); ax[j].set_xlabel(xlabel); ax[j].set_ylabel(ylabel)
            ax[j].set_xlim(xmin,xmax); ax[j].set_ylim(ymin,ymax)
            temp, =ax[j].plot(x,y[:,j,0],c=color); line.append(temp)
        def animate(i):
            for j in range(y.shape[1]): line[j].set_ydata(y[:,j,i])
            return line
    ani=anm.FuncAnimation(fig,animate,t.shape[0],interval=40.0/speed,blit=True)
    if(save): ani.save(FileName+'.gif')
    if(show): plt.show()

# vector t, NPoints by t.shape[0] matrix x & y & z
# (x[i,j],y[i,j],z[i,j]) is the coordinate of i-th point at t[j] time
# default FPS = 25
# PlotType can be chosen from scatter, surface
# for scatter:
#     color should be either a single string or a NPoints string list
# else:
#     If colormap is blank, will plot in single color as specified in color
def Animate3D(t, x, y, z,
title=' ', xlabel='x', ylabel='y', zlabel='z',
speed=1.0, save=False, FileName='3D', show=True,
PlotType='surface', color='black', colormap='', size=100):
    fig = plt.figure(); ax = plt.subplot(111, projection='3d')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    xmin = numpy.amin(x); xmax = numpy.amax(x); ax.set_xlim(xmin, xmax)
    ymin = numpy.amin(y); ymax = numpy.amax(y); ax.set_ylim(ymin, ymax)
    zmin = numpy.amin(z); zmax = numpy.amax(z); ax.set_zlim(zmin, zmax)
    if PlotType == 'scatter':
        temp=ax.scatter(x[:,0],y[:,0],z[:,0],s=size,color=color,depthshade=True); line=[temp]
        def animate(i):
            ax.clear()
            ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
            ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
            ax.scatter(x[:,i],y[:,i],z[:,i],s=size,color=color,depthshade=True)
            return line
    else:
        tol=0.001*abs(max(zmax,zmin)); indice=Pick_Significant(z[:,0],tol)# Neglect small value
        xplot=numpy.empty(len(indice)); yplot=numpy.empty(len(indice)); zplot=numpy.empty(len(indice))
        for j in range(len(indice)):
            xplot[j]=x[indice[j],0]; yplot[j]=y[indice[j],0]; zplot[j]=z[indice[j],0]
        if colormap=='': temp=ax.plot_trisurf(xplot,yplot,zplot,color=color)
        else: temp=ax.plot_trisurf(xplot,yplot,zplot,cmap=colormap,vmin=zmin,vmax=zmax)
        line=[temp]
        def animate(i):
            ax.clear()
            ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
            ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
            indice=Pick_Significant(z[:,i],tol)
            xplot=numpy.empty(len(indice)); yplot=numpy.empty(len(indice)); zplot=numpy.empty(len(indice))
            for j in range(len(indice)):
                xplot[j]=x[indice[j],i]; yplot[j]=y[indice[j],i]; zplot[j]=z[indice[j],i]
            if colormap=='': ax.plot_trisurf(xplot,yplot,zplot,color=color)
            else: ax.plot_trisurf(xplot,yplot,zplot,cmap=colormap,vmin=zmin,vmax=zmax)
            return line
    ani=anm.FuncAnimation(fig,animate,t.shape[0],interval=int(40.0/speed),blit=True)
    if(save): ani.save(FileName+'.gif')
    if(show): plt.show()

# matplotlib does not support multi-level 3D plot, because plt.subplot(*,projection='3d') returns
# a single Axes3DSubplot object rather than a list of objects

if __name__ == "__main__": print(__doc__)