#!/usr/bin/env python

__author__ = "Benjamin Quici"
__date__ = "02/03/2021"

import numpy as np
import pylab
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle

from astropy import wcs
from astropy.io import fits
from astropy.visualization.mpl_normalize import simple_norm

import warnings

from matplotlib.widgets import Slider, RadioButtons
from astropy.visualization import PercentileInterval

from astropy.coordinates import SkyCoord

import astropy.units as u

from sf.synchrofit import spectral_fitter, spectral_model, spectral_plotter, spectral_ages, spectral_units, logger, color_text, Colors

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

def within_circle(x, y, xc, yc, rad):
    if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < rad:
        return (True)
    else:
        return (False)

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

def gaussian2d(x, y, mux, muy, sigmax, sigmay, theta):
   a = np.cos(theta)**2 / (2*sigmax**2) + np.sin(theta)**2 / (2*sigmay**2)
   b = -np.sin(2*theta) / (4*sigmax**2) + np.sin(2*theta) / (4*sigmay**2)
   c = np.sin(theta)**2 / (2*sigmax**2) + np.cos(theta)**2 / (2*sigmay**2)
   g = np.exp(-(a*(x-mux)**2 + 2*b*(x-mux)*(y-muy) + c*(y-muy)**2))
   return(g)

def create_index_array(hdu, meerkat=False):
    # Initialise an array of co-ordinate pairs for the full array
    hdu[0].data = np.squeeze(hdu[0].data)
    if meerkat is True:
        imdata = np.squeeze(hdu[0].data)
        hdu[0].data = imdata[0]
    indexes = np.empty(((hdu[0].data.shape[0])*(hdu[0].data.shape[1]),2),dtype=int)
    idx = np.array([ (j,0) for j in range(hdu[0].data.shape[1])])
    j=hdu[0].data.shape[1]
    for i in range(hdu[0].data.shape[0]):
        idx[:,1]=i
        indexes[i*j:(i+1)*j] = idx
    # print(np.shape(indexes))
    return indexes

def astrofig(hdu):
    imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(hdu)
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_axes([0.34,0.09,0.65,0.9], projection=w)
    ax.imshow(imdata, cmap='inferno')
    ax.axis('off')
    return(fig, ax, imdata)

def hexgrid(hdu, theta):
    imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(hdu)
    beamvolume = (1.1331 * bmaj * bmin)   
    grid_x, grid_y = np.mgrid[0:naxis1:1, 0:naxis1:1] 
    indexes = create_index_array(hdu)
    # make and overlay the grid
    xc,yc = 0.5*naxis1, 0.5*naxis1  
    d_pix = bmaj/(pix2deg)
    xlength = int(naxis1/(d_pix))
    ylength = int(naxis1/(d_pix*(np.sqrt(3)/2)))
    xlist,ylist,beam_sum,x,y = [],[], [],[], []
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
            if within_circle(x1, y1, xc, yc, 60/(3600*pix2deg)) is True:
                g2d = gaussian2d(grid_x, grid_y, y1, x1, 0.5*d_pix, 0.5*d_pix,
                                    np.radians(180. - bpa))
                g = np.ravel(g2d)
                inside_beam = np.where(g > 0.1)
                flux_values = []
                for ix in indexes[inside_beam]:
                    flux_values.append(hdu[0].data[ix[1], ix[0]])
                pixel_sum = np.sum(flux_values)
                int_flux = pixel_sum*((pix2deg)**2)/(beamvolume)
                beam_sum.append(int_flux)
                xlist.append(x1)
                ylist.append(y1)
        a = -a
    ralist, declist = w.wcs_pix2world(xlist,ylist, 0)
    return(xlist, ylist, ralist, declist, beam_sum,d_pix)

def polyplot(image, sigma):
    hdu = fits.open(image)
    imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(hdu)
    fig, ax, imdata = astrofig(hdu)
    if 'l' in plt.rcParams['keymap.yscale']:
        plt.rcParams['keymap.yscale'].remove('l')
    pct = 99.0
    interval = PercentileInterval(pct)
    vmin, vmax = interval.get_limits(imdata)

    axmin = fig.add_axes([0.07, 0.9, 0.2, 0.02])
    axmax  = fig.add_axes([0.07, 0.8, 0.2, 0.02])
    axnsigma = fig.add_axes([0.07,0.7,0.2,0.02])
    axtheta  = fig.add_axes([0.07, 0.25, 0.2, 0.02])
    axshift  = fig.add_axes([0.07, 0.15, 0.2, 0.02])
    
    svmin = Slider(axmin, "vmin \n (Jy/beam)", vmin, vmax, valinit=3*sigma)
    svmax = Slider(axmax, "vmax \n (Jy/beam)", vmin, vmax, valinit=vmax)
    svnsigma = Slider(axnsigma, "nsigma", 1, 20, valinit=3)
    svtheta = Slider(axtheta, "theta \n (deg)", 0, 60, valinit=0)
    svshift = Slider(axshift, "shift", 0, 1, valinit=0)
    
    def update(val):
        ax.cla()
        xcoords, ycoords, racoords, deccoords, beam_sum, d_pix = hexgrid(hdu, svtheta.val)
        ax.imshow(imdata, cmap='inferno',vmin=svmin.val, vmax=svmax.val)
        ax.contour(imdata, levels=svnsigma.val*np.array([sigma]), colors='white')
        file = open('test.dat', "w")
        file.write("ralist, declist, flux \n")
        for i in range(0,len(xcoords)):
            circle = Circle((xcoords[i], ycoords[i]), radius=0.5*d_pix, edgecolor='white', facecolor='None', lw=1)
            ax.add_patch(circle)

            file.write("{}, {}, {} \n".format(racoords[i], deccoords[i], beam_sum[i]))
        file.close()
        fig.canvas.draw_idle()
    
    svmin.on_changed(update)
    svmax.on_changed(update)
    svtheta.on_changed(update)
    svnsigma.on_changed(update)
    svshift.on_changed(update)

    polypick = PolyPick(ax)
    plt.show()
    
    df_data = pd.read_csv('test.dat')
    racoords = (df_data.iloc[:,0].values)
    deccoords = (df_data.iloc[:,1].values)
    flux = (df_data.iloc[:,2].values)
    lobe1_selected_coords = Coords()
    lobe2_selected_coords = Coords()
    lobe1_selected_coords.x, lobe1_selected_coords.y = w.wcs_pix2world(polypick.lobe1.x,polypick.lobe1.y, 0)
    lobe2_selected_coords.x, lobe2_selected_coords.y = w.wcs_pix2world(polypick.lobe2.x,polypick.lobe2.y, 0)
    hexgrid_coords = SkyCoord(racoords, deccoords, unit=u.deg)
    lobe1_coords = SkyCoord(lobe1_selected_coords.x, lobe1_selected_coords.y, unit=u.deg)
    lobe2_coords = SkyCoord(lobe2_selected_coords.x, lobe2_selected_coords.y, unit=u.deg)
    idx_l1, d2d_l1, d3d_l1 = lobe1_coords.match_to_catalog_sky(hexgrid_coords)
    idx_l2, d2d_l2, d3d_l2 = lobe2_coords.match_to_catalog_sky(hexgrid_coords)


    lobe1_hex_coords = Coords()
    lobe2_hex_coords = Coords()
    lobe1_coords = (hexgrid_coords[idx_l1])
    lobe2_coords = (hexgrid_coords[idx_l2])
    lobe1_hex_coords.x, lobe1_hex_coords.y = w.wcs_world2pix(lobe1_coords.ra.degree, lobe1_coords.dec.degree, 0)
    lobe2_hex_coords.x, lobe2_hex_coords.y = w.wcs_world2pix(lobe2_coords.ra.degree, lobe2_coords.dec.degree, 0)

    lobe1_flux = flux[idx_l1]
    lobe2_flux = flux[idx_l2]

    fig, ax, imdata = astrofig(hdu)
    for i in range(0,len(lobe1_hex_coords.x)):
        circle = Circle((lobe1_hex_coords.x[i], lobe1_hex_coords.y[i]), radius=0.5*bmaj/(pix2deg), edgecolor='cyan', facecolor='None', lw=1)
        ax.add_patch(circle)
    for i in range(0,len(lobe2_hex_coords.x)):
        circle = Circle((lobe2_hex_coords.x[i], lobe2_hex_coords.y[i]), radius=0.5*bmaj/(pix2deg), edgecolor='magenta', facecolor='None', lw=1)
        ax.add_patch(circle)
    plt.show()

    return(lobe1_hex_coords, lobe2_hex_coords, lobe1_flux, lobe2_flux)

# image = '/home/sputnik/Documents/Thesis/Paper_III/MJ225337-34/MJ225337-34_1.3GHz_cv_10.1arcsec.fits'
image='/home/sputnik/Documents/Thesis/Paper_III/MJ225337-34/J23-34_r1_2.5_2.5_spw4~7.image.tt0.pb_cv_10.1arcsec_rg.fits'
l1coord,l2coord,s1,s2 = polyplot(image, 0.0001)
print(test)
images = ['MJ225337-34_375MHz_cv_10.1arcsec.fits','MJ225337-34_682MHz_cv_10.1arcsec.fits','MJ225337-34_887MHz_cv_10.1arcsec.fits',\
    'MJ225337-34_1.3GHz_cv_10.1arcsec.fits','J23-34_r1_2.5_2.5_spw4~7.image.tt0.pb_cv_10.1arcsec_rg.fits','J23-34_r0.5_3.5_3.5_spw0~3.image.tt0.pb_cv_10.1arcsec_rg.fits']
resolved_sed_l1 = []
resolved_sed_l2 = []
freq_l1 = []
freq_l2 = []
freq = [375, 682, 887, 1283.89, 4989, 6016]
beamcoords = Coords()
for ii in range(0,len(images)):
    hdu = fits.open('/home/sputnik/Documents/Thesis/Paper_III/MJ225337-34/'+images[ii])
    imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(hdu)
    beamvolume = (1.1331 * bmaj * bmin)   
    grid_x, grid_y = np.mgrid[0:naxis1:1, 0:naxis1:1] 
    indexes = create_index_array(hdu)
    # make and overlay the grid
    xc,yc = 0.5*naxis1, 0.5*naxis1  
    d_pix = bmaj/(pix2deg)
    xlist,ylist,beam_sum,x,y = [],[], [],[], []
    count=0
    for jj in [l1coord, l2coord]:
        beam_sum = []
        count = count+1
        # print(count)
        ra, dec = jj.x, jj.y
        for i in range(0,len(ra)):
            x1, y1 = ra[i], dec[i]
            if ii == 0:
                beamcoords.x.append(x1)
                beamcoords.y.append(y1)
            g2d = gaussian2d(grid_x, grid_y, y1, x1, 0.5*d_pix, 0.5*d_pix,
                                np.radians(180. - bpa))
            g = np.ravel(g2d)
            inside_beam = np.where(g > 0.1)
            flux_values = []
            for ix in indexes[inside_beam]:
                flux_values.append(hdu[0].data[ix[1], ix[0]])
            pixel_sum = np.sum(flux_values)
            int_flux = pixel_sum*((pix2deg)**2)/(beamvolume)
            beam_sum.append(int_flux)
        if count==1:
            # print(images[ii], count)
            resolved_sed_l1.append(beam_sum)
            freq_l1.append(freq[ii])
        elif count==2:
            # print(images[ii], count)
            resolved_sed_l2.append(beam_sum)
            freq_l2.append(freq[ii])

resolved_sed_l1 = (np.asarray(resolved_sed_l1))
resolved_sed_l1 = resolved_sed_l1.T

resolved_sed_l2 = (np.asarray(resolved_sed_l2))
resolved_sed_l2 = resolved_sed_l2.T

fullsed = (np.concatenate([resolved_sed_l1, resolved_sed_l2]))

fig = plt.figure(figsize=(15,10))
cm = pylab.get_cmap('rainbow')
axl1 = fig.add_axes([0.05,0.5,0.45,0.45])
axl2 = fig.add_axes([0.05,0.05,0.45,0.45])
for ax in [axl1, axl2]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(300,10000)
axl1.set_xlabel('Frequency')
axl1.set_ylabel('Flux density')
axl2.set_ylabel('Flux density')
for i in range(0,len(fullsed)):
    color = cm(1.*i/len(fullsed))
    if i <= len(resolved_sed_l1)-1:
        axl1.scatter([375, 682, 887, 1283.89, 4989, 6016],fullsed[i], marker='.', color=color)
        # axl1.errorbar([375, 682, 887, 1283.89, 4989, 6016],fullsed[i],yerr=0.03*fullsed[i], xerr=0, color=color)
        params = spectral_fitter(1e+6*np.array([375, 682, 887, 1283.89, 4989, 6016]), fullsed[i], 0.03*fullsed[i], 'JP')
        model_data = spectral_model(params, np.geomspace(1e+8,1e+10,100))[0]
        axl1.plot(np.geomspace(1e+2,1e+4,100), model_data, color=color)
    else:
        axl2.scatter([375, 682, 887, 1283.89, 4989, 6016],fullsed[i], marker='.', color=color)
        # axl2.errorbar([375, 682, 887, 1283.89, 4989, 6016],fullsed[i],yerr=0.03*fullsed[i], xerr=0, color=color)
        params = spectral_fitter(1e+6*np.array([375, 682, 887, 1283.89, 4989, 6016]), fullsed[i], 0.03*fullsed[i], 'JP')
        model_data = spectral_model(params, np.geomspace(1e+8,1e+10,100))[0]
        axl2.plot(np.geomspace(1e+2,1e+4,100), model_data, color=color)

imdata, header, w, bmaj, bmin, bpa, pix2deg, naxis1 = get_hdu_info(fits.open('/home/sputnik/Documents/Thesis/Paper_III/MJ225337-34/MJ225337-34_1.3GHz_cv_10.1arcsec.fits'))
ax = fig.add_axes([0.45,0.175,0.65,0.65], projection=w)
ax.imshow(imdata, cmap='inferno')
for i in range(0,len(beamcoords.x)):
    color = cm(1.*i/len(beamcoords.x))
    circle = Circle((beamcoords.x[i], beamcoords.y[i]), radius=0.5*bmaj/(pix2deg), edgecolor=color, facecolor='None', lw=1)
    ax.add_patch(circle)
plt.show()

#33.23