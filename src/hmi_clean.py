# {{{ Library imports
from pyshtools import legendre as pleg
from sunpy.coordinates import frames
from sunpy.map import Map as spMap
from globalvars import DopplerVars
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from math import pi
import numpy as np
import argparse
import time
import os
# }}} imports

__all__ = ["get_pleg_index",
           "gen_leg",
           "gen_leg_x",
           "inv_SVD",
           "HmiClass"]
__author__ "Samarth G Kashyap"

# {{{ def get_pleg_index(l, m):
def get_pleg_index(l, m):
    """Gets the index for accessing legendre polynomials
    (generated from pyshtools.legendre)

    Parameters:
    -----------
    l : int
        Spherical Harmonic degree
    m : int
        Azimuthal order

    Returns:
    --------
    int
        index for accessing legendre polynomial
    """
    return int(l*(l+1)/2 + m)
# }}} get_pleg_index(l, m):


# {{{ def gen_leg(lmax, theta):
def gen_leg(lmax, theta):
    """Generates associated legendre polynomials and derivatives (normalized)

    Parameters:
    -----------
    lmax : int
        Maximum spherical harmonic degree
    theta : np.ndarray(ndim=1, dtype=np.float64)
        1D array containing theta for computing P_l(cos(theta))

    Returns:
    --------
    (leg, leg_d1) : list of np.ndarray(ndim=2)
        Legendre polynomials and it's derivatives
    """
    cost = np.cos(theta)
    sint = np.sin(theta).reshape(1, theta.shape[0])

    maxIndex = int(lmax+1)
    ell = np.arange(maxIndex)
    norm = np.sqrt(ell*(ell+1)).reshape(maxIndex, 1)
    norm[norm == 0] = 1

    leg = np.zeros((maxIndex, theta.size))
    leg_d1 = np.zeros((maxIndex, theta.size))

    count = 0
    for z in cost:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2)/norm, leg_d1 * (-sint)/np.sqrt(2)/norm
# }}} gen_leg(lmax, theta)


# {{{ def gen_leg_x(lmax, x):
def gen_leg_x(lmax, x):
    """Generates associated legendre polynomials and derivatives
    for a given x (normalized)

    Parameters:
    -----------
    lmax : int
        Maximum spherical harmonic degree
    x: float
       x for computing P_l(x)

    Returns:
    --------
    (leg, leg_d1) : list
        Legendre polynomial and it's derivative
    """
    maxIndex = int(lmax+1)
    ell = np.arange(maxIndex)
    norm = np.sqrt(ell*(ell+1)).reshape(maxIndex, 1)
    norm[norm == 0] = 1

    leg = np.zeros((maxIndex, x.size))
    leg_d1 = np.zeros((maxIndex, x.size))

    count = 0
    for z in x:
        leg[:, count], leg_d1[:, count] = pleg.PlBar_d1(lmax, z)
        count += 1
    return leg/np.sqrt(2)/norm, leg_d1/np.sqrt(2)/norm
# }}} gen_leg_x(lmax, x)


# {{{ def inv_SVD(A, svdlim):
def inv_SVD(A, svdlim):
    u, s, v = np.linalg.svd(A, full_matrices=False)
    sinv = s**-1
    sinv[sinv/sinv[0] > svdlim] = 0.0  # svdlim
    return np.dot(v.transpose().conjugate(),
                  np.dot(np.diag(sinv), u.transpose().conjugate()))
# }}} inv_SVD(A, svdlim)


class HmiClass():
    """Class to handle mi maps and their coordinates"""
    # {{{ def __init__(self, hmi_data_dir, hmi_file, day):
    def __init__(self, hmi_data_dir, hmi_file, day):
        print(f"Loading {hmi_file}")
        self.day = day
        self.rsun_rel = 200

        self.fname = hmi_data_dir + hmi_file
        hmi_map = spMap(self.fname)
        self.rsun_meters = hmi_map.rsun_meters
        self.B0 = hmi_map.observer_coordinate.lat
        self.P0 = hmi_map.observer_coordinate.lon

        x, y = np.meshgrid(*[np.arange(v.value) for v in hmi_map.dimensions])\
            * u.pix
        hpc_coords = hmi_map.pixel_to_world(x, y)
        r = np.sqrt(hpc_coords.Tx ** 2 +
                    hpc_coords.Ty ** 2) / hmi_map.rsun_obs
        rcrop = 0.95
        mask_r = r > rcrop

        hmi_map.data[mask_r] = np.nan
        r[mask_r] = np.nan

        hpc_hgf = hpc_coords.transform_to(frames.HeliographicStonyhurst)
        hpc_hc = hpc_coords.transform_to(frames.Heliocentric)

        self.lat = hpc_hgf.lat
        self.lon = hpc_hgf.lon
        self.coords_r = r.copy()
        self.coords_hc_x = hpc_hc.x.copy()
        self.coords_hc_y = hpc_hc.y.copy()
        self.map_data = hmi_map.data.copy()
        self.mask_nan = ~np.isnan(self.map_data)
        return None
    # }}} __init__(self, hmi_data_dir, hmi_file, day)

    # {{{ def save_theta_phi_DC(self):
    def save_theta_phi_DC(self):
        """Saves coordinate of HMI map (ref: disc center)"""
        rho = np.sqrt(self.coords_hc_x**2 + self.coords_hc_y**2)
        psi = np.arctan2(self.coords_hc_y, self.coords_hc_x)
        ph = np.zeros(psi.shape) * u.rad

        # range of ph; 0 < ph < 2\pi
        ph[psi < 0] = psi[psi < 0] + (2*pi)*u.rad
        ph[~(psi < 0)] = psi[~(psi < 0)]
        th = np.arcsin(rho/self.rsun_meters)
        print(f"Writing {gvar.outdir}thDC_{gvar.year}_{self.day:03d}.npy")
        np.save(f"{gvar.outdir}thDC_{gvar.year}_{self.day:03d}.npy", th.value)
        print(f"Writing {gvar.outdir}phDC_{gvar.year}_{self.day:03d}.npy")
        np.save(f"{gvar.outdir}phDC_{gvar.year}_{self.day:03d}.npy", ph.value)
        return None
    # }}} save_theta_phi_DC(self)

    # {{{ def save_theta_phi(self):
    def save_theta_phi(self):
        """Saves the coordinates of HMI map"""
        lat = (self.lat + 90*u.deg).value
        lon = (self.lon).value
        print(f"Writing {gvar.outdir}th_{gvar.year}_{self.day:03d}.npy")
        np.save(f"{gvar.outdir}th_{gvar.year}_{self.day:03d}.npy", lat)
        print(f"Writing {gvar.outdir}ph_{gvar.year}_{self.day:03d}.npy")
        np.save(f"{gvar.outdir}ph_{gvar.year}_{self.day:03d}.npy", lon)
        return None
    # }}} save_theta_phi(self)

    # {{{ save_map_data(self)
    def save_map_data(self):
        """Saves image data of HMI map"""
        print(f"Writing {gvar.outdir}residual_{gvar.year}_{self.day:03d}.npy")
        np.save(f"{gvar.outdir}residual_{gvar.year}_{self.day:03d}.npy",
                self.map_data)
        return None
    # }}} save_map_data(self)

    # {{{ def get_sat_vel(self):
    def get_sat_vel(self):
        """Obtain the velocity components of the satellite
        by reading the FITS header file."""
        map_fits = fits.open(self.fname)
        map_fits.verify('fix')
        vx = map_fits[1].header['OBS_VR']
        vz = map_fits[1].header['OBS_VN']
        vy = map_fits[1].header['OBS_VW']
        self.sat_VR, self.sat_VN, self.sat_VW = vx, vz, vy
        self.sat_VX, self.sat_VY, self.sat_VZ = vx, vy, vz
        print(f"VR = {self.sat_VR}, VN = {self.sat_VN}, VW = {self.sat_VW}")
        return None
    # }}} get_sat_vel(self)

    # {{{ def remove_grav_redshift(self):
    def remove_grav_redshift(self):
        """Removing the effect of gravitation red-shift. It is a 
        DC shift between the solar surface and the observer"""
        self.map_data -= 632
    # }}} remove_grav_redshift(self)

    # {{{ def remove_sat_vel(self, method):
    def remove_sat_vel(self, method=2):
        """Removing the satellite velocity from the observed image.
        This is computed using two different methods and both 
        give the same result (nearly). 
        """
        self.get_sat_vel()
        if method == 1:
            thHG1 = self.lat[self.mask_nan]
            phHG1 = self.lon[self.mask_nan]

            ct, st = np.cos(thHG1), np.sin(thHG1)
            cp, sp = np.cos(phHG1), np.sin(phHG1)

            # getting velocity components in (r, theta, phi) directions
            # given by (vr1, vt1, vp1)
            vr1 = cp*st*self.sat_VX + sp*st*self.sat_VY + ct*self.sat_VZ
            vt1 = cp*ct*self.sat_VX + sp*ct*self.sat_VY - st*self.sat_VZ
            vp1 = -sp*self.sat_VX + cp*self.sat_VY

            # Getting the line of sight vector (lr, lt, lp)
            sB0 = np.sin(self.B0)
            cB0 = np.cos(self.B0)
            lr = sB0*ct + cB0*st*cp
            lt = sB0*st - cB0*ct*cp
            lp = cB0*sp

            # the LoS velocity = l \cdot v
            vC1 = lr*vr1 + lt*vt1 + lp*vp1
            velCorr = np.zeros((4096, 4096))
            velCorr[self.mask_nan] = vC1
            velCorr[~self.mask_nan] = np.nan

        elif method == 2:
            sigma = np.arctan(self.coords_r/self.rsun_rel)
            chi = np.arctan2(self.coords_hc_x, self.coords_hc_y)
            ssig, csig = np.sin(sigma), np.cos(sigma)
            schi, cchi = np.sin(chi), np.cos(chi)

            vr1 = self.sat_VR*csig
            vr2 = -self.sat_VW*ssig*schi
            vr3 = -self.sat_VN*ssig*cchi
            vC1 = vr1 + vr2 + vr3
            velCorr = np.zeros((4096, 4096))
            velCorr[self.mask_nan] = vC1[self.mask_nan]
            velCorr[~self.mask_nan] = np.nan

        self.map_data += velCorr
        return None
    # }}} remove_sat_vel(self, method)

    # {{{ def remove_large_features(self):
    def remove_large_features(self):
        """Removal of differential rotation, meridional circulation
        and limb-shift from the image.
        """
        lat = self.lat + 90*u.deg
        lon = self.lon
        cB0 = np.cos(self.B0)
        sB0 = np.sin(self.B0)

        lat1D = lat[self.mask_nan].copy()
        lon1D = lon[self.mask_nan].copy()
        rho1D = self.coords_r[self.mask_nan].copy()
        ct, st = np.cos(lat1D), np.sin(lat1D)
        cp, sp = np.cos(lon1D), np.sin(lon1D)

        print("-- Generating Legendre polynomials")
        t1 = time.time()
        pl_theta, dt_pl_theta = gen_leg(5, lat1D)
        t2 = time.time()
        print(f"--- Time taken for pl_theta = {(t2 - t1)/60} minutes")
        pl_rho, dt_pl_rho = gen_leg_x(5, rho1D)
        t3 = time.time()
        print(f"--- Time taken for pl_rho = {(t3 - t2)/60} minutes")

        lt = sB0 * st - cB0 * ct * cp
        lp = cB0 * sp
        im_arr = np.zeros((11, lt.shape[0]))
        A = np.zeros((11, 11))
        # differential rotation (axisymmetric feature; s = 1, 3, 5)
        im_arr[0, :] = dt_pl_theta[1, :] * lp
        im_arr[1, :] = dt_pl_theta[3, :] * lp
        im_arr[2, :] = dt_pl_theta[5, :] * lp

        # meridional circulation (axisymmetric feature; s = 2, 4)
        im_arr[3, :] = dt_pl_theta[2, :] * lt
        im_arr[4, :] = dt_pl_theta[4, :] * lt

        # limb-shift
        # axisymmetric feature (frame=pole at disk-center)
        # s = 0-5
        im_arr[5, :] = pl_rho[0, :]
        im_arr[6, :] = pl_rho[1, :]
        im_arr[7, :] = pl_rho[2, :]
        im_arr[8, :] = pl_rho[3, :]
        im_arr[9, :] = pl_rho[4, :]
        im_arr[10, :] = pl_rho[5, :]

        mapArr = self.map_data[self.mask_nan].copy()

        RHS = im_arr.dot(mapArr)

        for i in range(11):
            for j in range(11):
                A[i, j] = im_arr[i, :].dot(im_arr[j, :])

        Ainv = inv_SVD(A, 1e5)
        fit_params = Ainv.dot(RHS)
        print(f"Rotation = {fit_params[:3]} m/s,"
              + f"\nMeridional Circ = {fit_params[3:5]} m/s,"
              + f"\nLimb Shift = {fit_params[5:]} m/s\n")
        print(f"Rotation = {fit_params[:3]/2/pi/0.695} Hz")
        new_img_arr = fit_params.dot(im_arr)

        # getting individual feature maps
        # diff_rot = fit_params[:3].dot(im_arr[:3, :])
        # meridional_circ = fit_params[3:5].dot(im_arr[3:5, :])
        # limb_shift = fit_params[5:].dot(im_arr[5:, :])

        self.map_data[self.mask_nan] -= new_img_arr
        return None
    # }}} remove_large_features(self)
