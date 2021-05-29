#!/usr/bin/env python

__author__ = "Benjamin Quici"
__date__ = "02/03/2021"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle

from astropy import wcs
from astropy.io import fits
from astropy.visualization.mpl_normalize import simple_norm

import warnings

from matplotlib.widgets import Slider, RadioButtons
from astropy.visualization import PercentileInterval

warnings.filterwarnings('ignore')

lobe1 = {"marker": ".", "linestyle": "None", "color": "cyan", "s":100}
lobe2 = {"marker": ".", "linestyle": "None", "color": "magenta", "s":100}
measurementmarker = {"marker": ".", "linestyle": "None", "color": "white", "s":100}
limitmarker = {"marker": "v", "linestyle": "None", "facecolor": "None", "edgecolor":"white", "s":100}

class vector:
    def __init__(self, x=0, y=0):
        self.i = x
        self.j = y

class Coords:
    def __init__(self):
        self.x = []
        self.y = []

class PolyPick:
    def __init__(self, ax=None):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self)
        self.measurement = Coords()
        self.limit = Coords()
        self.lobe1 = Coords()
        self.lobe2 = Coords()

    def __call__(self, event):
        if event.inaxes != self.ax.axes: return
        if event.key == 'm':
            self.measurement.x.append(event.xdata)
            self.measurement.y.append(event.ydata)
        elif event.key == 'l':
            self.limit.x.append(event.xdata)
            self.limit.y.append(event.ydata)
        elif event.key == '1':
            self.lobe1.x.append(event.xdata)
            self.lobe1.y.append(event.ydata)
        elif event.key == '2':
            self.lobe2.x.append(event.xdata)
            self.lobe2.y.append(event.ydata)

        # This draws points onto the canvas
        self.ax.scatter(self.measurement.x, self.measurement.y, **measurementmarker)
        self.ax.scatter(self.limit.x, self.limit.y, **limitmarker)
        self.ax.scatter(self.lobe1.x, self.lobe1.y, **lobe1)
        self.ax.scatter(self.lobe2.x, self.lobe2.y, **lobe2)

        # This makes it plot without needing to change focus back to the terminal
        self.ax.figure.canvas.draw()

def translate(shift):
    vec = vector()
    vec.i, vec.j = shift, 0
    return(vec)

def nn_vector(d):
    vec = vector()
    vec.i, vec.j  = d, 0
    return(vec)

def nn2_vector(d, a):
    vec = vector()
    vec.i, vec.j = 0.25*d*a, 0.5*np.sqrt(3)*d
    return(vec)

def rotation_matrix(theta):
    theta = theta * np.pi / 180.
    rotation_matrix = np.zeros(4)
    rotation_matrix = rotation_matrix.reshape(2, 2)
    rotation_matrix[0, 0] = np.cos(theta)
    rotation_matrix[0, 1] = -np.sin(theta)
    rotation_matrix[1, 0] = np.sin(theta)
    rotation_matrix[1, 1] = np.cos(theta)
    return(rotation_matrix)

def get_hdu_info(hdu):
    header = hdu[0].header
    w = wcs.WCS(header, naxis=2)
    imdata = np.squeeze(hdu[0].data)
    bmaj = header['BMAJ']
    bmin = header['BMIN']
    bpa = header['BPA']
    pix2deg = header['CDELT2']
    naxis1 = header['NAXIS1']
    return(imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1)

def polyplot():
    hdu = fits.open('/home/sputnik/Documents/Thesis/Paper_III/MJ225337-34/MJ225337-34_682MHz_cv_10.1arcsec.fits')
    imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(hdu)

    # make and overlay the grid
    theta=0
    xc,yc = 0.5*naxis1, 0.5*naxis1  
    d = 10.1
    d_pix = bmaj/(pix2deg)
    xlength = int(naxis1/(d_pix))
    ylength = int(naxis1/(d_pix*(np.sqrt(3)/2)))
    xlist,ylist = [],[]
    a = 1
    for j in range(0,ylength):
        for i in range(0,xlength):
            rotmat = rotation_matrix(theta)
            rotmat_ = np.matmul(rotmat, [1, 1])
            xc_offset = 0.5*naxis1 * (-rotmat_[0]) + xc 
            yc_offset = 0.5*naxis1 * (-rotmat_[1]) + yc
            x_0 = i*nn_vector(d_pix).i + nn2_vector(d_pix, a).i + translate(0.5*d_pix).i
            y_0 = i*nn_vector(d_pix).j + j *nn2_vector(d_pix, a).j
            coords_0 = np.array([x_0, y_0])
            coords_1 = np.matmul(rotmat, coords_0)
            x1,y1 = coords_1[0]+ xc_offset, coords_1[1]+ yc_offset
            xlist.append(x1)
            ylist.append(y1)
        a = -a
        
    pct = 99.0
    interval = PercentileInterval(pct)
    vmin, vmax = interval.get_limits(imdata)

    def update(val):
        img.set_clim([svmin.val, svmax.val])
        a = 1
        for j in range(0,ylength):
            for i in range(0,xlength):
                rotmat = rotation_matrix(val)
                rotmat_ = np.matmul(rotmat, [1, 1])
                xc_offset = 0.5*naxis1 * (-rotmat_[0]) + xc 
                yc_offset = 0.5*naxis1 * (-rotmat_[1]) + yc
                x_0 = i*nn_vector(d_pix).i + nn2_vector(d_pix, a).i + translate(0.5*d_pix).i
                y_0 = i*nn_vector(d_pix).j + j *nn2_vector(d_pix, a).j
                coords_0 = np.array([x_0, y_0])
                coords_1 = np.matmul(rotmat, coords_0)
                x1,y1 = coords_1[0]+ xc_offset, coords_1[1]+ yc_offset
                xlist.append(x1)
                ylist.append(y1)
            a = -a
        for i in range(0,len(xlist)):
            circle = Circle((xlist[i], ylist[i]), radius=0.5*d_pix, edgecolor='white', facecolor='None', lw=1)
            ax.add_patch(circle)
        fig = plt.figure(figsize=(10,10))
        if 'l' in plt.rcParams['keymap.yscale']:
            plt.rcParams['keymap.yscale'].remove('l')
        ax = fig.add_axes([0.1,0.1,0.89,0.89], projection=w)
        axmin = fig.add_axes([0.05, 0.05, 0.2, 0.02])
        axmax  = fig.add_axes([0.44, 0.05, 0.2, 0.02])
        axtheta  = fig.add_axes([0.74, 0.05, 0.2, 0.02])
        norm = simple_norm(imdata,percent=99.5)
        img = ax.imshow(imdata, norm=norm, cmap='inferno')
        for i in range(0,len(xlist)):
            circle = Circle((xlist[i], ylist[i]), radius=0.5*d_pix, edgecolor='white', facecolor='None', lw=1)
            ax.add_patch(circle)
        ax.contour(imdata, levels=np.array([0.000417]),colors='white')
        fig.canvas.draw()

    fig = plt.figure(figsize=(10,10))
    # if 'l' in plt.rcParams['keymap.yscale']:
    #     plt.rcParams['keymap.yscale'].remove('l')
    ax = fig.add_axes([0.1,0.1,0.89,0.89], projection=w)
    axmin = fig.add_axes([0.05, 0.05, 0.2, 0.02])
    axmax  = fig.add_axes([0.44, 0.05, 0.2, 0.02])
    axtheta  = fig.add_axes([0.74, 0.05, 0.2, 0.02])
    # norm = simple_norm(imdata,percent=99.5)
    # img = ax.imshow(imdata, norm=norm, cmap='inferno')
    # for i in range(0,len(xlist)):
    #     circle = Circle((xlist[i], ylist[i]), radius=0.5*d_pix, edgecolor='white', facecolor='None', lw=1)
    #     ax.add_patch(circle)
    # ax.contour(imdata, levels=np.array([0.000417]),colors='white')

    svmin = Slider(axmin, "vmin", vmin, vmax, valinit=3*0.0001339)
    svmax = Slider(axmax, "vmax", vmin, vmax, valinit=4*0.0001339)
    svtheta = Slider(axtheta, "theta", 0, 60, valinit=0)

    svmin.on_changed(update)
    svmax.on_changed(update)
    print(svtheta.on_changed(update))


    polypick = PolyPick(ax)
    plt.show()

polyplot()
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_axes([0.1,0.1,0.89,0.89], projection=w)
# norm = simple_norm(imdata,percent=99.5)
# ax.imshow(imdata, norm=norm, cmap='hot')
# ax.scatter(polypick.lobe1.x,polypick.lobe1.y, c='cyan', s=1)
# ax.scatter(polypick.lobe2.x,polypick.lobe2.y, c='magenta', s=1)
# plt.show()