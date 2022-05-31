#from penquins import Kowalski
import numpy as np
#from concurrent.futures import as_completed
#import pandas as pd
#import tables
#import math
from astropy.io import fits
from astropy.io.fits import getdata
import glob
from glob import glob
import sys

#import KowalskiQuery
import os

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt 

import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')


from matplotlib import rc, font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter 
from matplotlib import gridspec


from random import random
import uuid

projects_dir = os.path.dirname(os.path.abspath(__file__))

print (projects_dir)


    


def load_templates(template_list):
    templates = np.zeros((500, template_list.size))
    for i, fname in enumerate(template_list):
        aux = getdata(fname)
        templates[:, i] = aux
    return templates
    

phase_model = np.arange(500)*0.002

template_list_u = np.concatenate((np.sort(glob("%s/bspline_templates/1??u.fits" % projects_dir)), np.sort(glob("%s/bspline_templates/?u.fits" % projects_dir))))
templates_u = load_templates(template_list_u)
template_list_g = np.concatenate((np.sort(glob("%s/bspline_templates/1??g.fits" % projects_dir)), np.sort(glob("%s/bspline_templates/?g.fits" % projects_dir))))
templates_g = load_templates(template_list_g)
template_list_r = np.concatenate((np.sort(glob("%s/bspline_templates/1??r.fits" % projects_dir)), np.sort(glob("%s/bspline_templates/?r.fits" % projects_dir))))
templates_r = load_templates(template_list_r)
template_list_i = np.concatenate((np.sort(glob("%s/bspline_templates/1??i.fits" % projects_dir)), np.sort(glob("%s/bspline_templates/?i.fits" % projects_dir))))
templates_i = load_templates(template_list_i)
template_list_z = np.concatenate((np.sort(glob("%s/bspline_templates/1??z.fits" % projects_dir)), np.sort(glob("%s/bspline_templates/?z.fits" % projects_dir))))
templates_z = load_templates(template_list_z)


def model(ind,true_params):
	
    # model parameters
    period = true_params['P'][ind]
    uA = true_params['uA'][ind]
    gA = true_params['gA'][ind]
    rA = true_params['rA'][ind]
    iA = true_params['iA'][ind]
    zA = true_params['zA'][ind]
    
    u0 = true_params['u0'][ind]
    g0 = true_params['g0'][ind]
    r0 = true_params['r0'][ind]
    i0 = true_params['i0'][ind]
    z0 = true_params['z0'][ind]
    
    if true_params['gT'][ind] > 99:
        template_u = templates_u[:, true_params['uT'][ind] - 100]
        template_g = templates_g[:, true_params['gT'][ind] - 100]
        template_r = templates_r[:, true_params['rT'][ind] - 100]
        template_i = templates_i[:, true_params['iT'][ind] - 100]
        template_z = templates_z[:, true_params['zT'][ind] - 100]
        
    else:

	# the offset was always -2 in original code by B. Sesar, but this is wrong for u and z band
        template_u = templates_u[:, -1 + true_params['uT'][ind]]
        template_g = templates_g[:, -2 + true_params['gT'][ind]]
        template_r = templates_r[:, -2 + true_params['rT'][ind]]
        template_i = templates_i[:, -2 + true_params['iT'][ind]]
        template_z = templates_z[:, -1 + true_params['zT'][ind]]
   
   
    return period, uA, gA, rA, iA, zA, u0, g0, r0, i0, z0, template_u, template_g, template_r, template_i, template_z


def main():
		
	
	# flag indicates if plots should be generated
	plot_flag = True
	
	# input list true_rrlyrae:
			#P		period
			#rHJD0		phase offset
			#uT		u band template
			#gT		g band template
			#rT		r band template
			#iT		i band template
			#zT		z band template
			#uA		u band mean amplitude
			#gA		g band amplitude
			#rA		r band amplitude
			#iA		i band amplitude
			#zA		z band amplitude
			#u0		u band mag
			#g0		g band mag
			#r0		r band mag
			#i0		i band mag
			#z0		z band mag
	
	# this input list is based on RR Lyrae from SDSS S82, Sesar+2010
	# a similar input list can be created by the user
	
	true_params = np.genfromtxt('true_rrlyrae.csv', delimiter=',', names='id, type, P, uA, u0, uT, gA, g0, gT, rA, r0, rT, iA, i0, iT, zA, z0, zT',  
				      dtype='u4, |U10, f8, f8, f8, u2, f8, f8, u2, f8, f8, u2, f8, f8, u2, f8, f8, u2')
	

	cadence_u_filename = 'cadence_templates/1/u_baseline_2snaps_v1.5_10yrs.db.dat'
	cadence_g_filename = 'cadence_templates/1/g_baseline_2snaps_v1.5_10yrs.db.dat'
	cadence_r_filename = 'cadence_templates/1/r_baseline_2snaps_v1.5_10yrs.db.dat'
	cadence_i_filename = ''
	cadence_z_filename = 'cadence_templates/1/z_baseline_2snaps_v1.5_10yrs.db.dat'
	
	
	cadence_u = []
	cadence_g = []
	cadence_r = []
	cadence_i = []
	cadence_z = []
	
	
	if(cadence_u_filename!=''):
		cadence_u =np.array(np.genfromtxt(cadence_u_filename, names='t',
				      dtype='f8')['t'])
	
	if(cadence_g_filename!=''):
		cadence_g = np.array(np.genfromtxt(cadence_g_filename, names='t',  
				      dtype='f8')['t'])
		
	if(cadence_r_filename!=''):
		cadence_r =np.array(np.genfromtxt(cadence_r_filename, names='t',  
				      dtype='f8')['t'])
	if(cadence_i_filename!=''):
		cadence_i =np.array(np.genfromtxt(cadence_i_filename, names='t',  
				      dtype='f8')['t'])	
	if(cadence_z_filename!=''):
		cadence_z =np.array(np.genfromtxt(cadence_z_filename, names='t',  
				      dtype='f8')['t'])

	if(plot_flag == True):
		
		t_start = min(np.concatenate((cadence_u,cadence_g,cadence_r,cadence_i,cadence_z)))
		t_end = max(np.concatenate((cadence_u,cadence_g,cadence_r,cadence_i,cadence_z)))
		observation_baseline = np.arange(np.floor(t_start),np.ceil(t_end),0.001)
	
	
	rHJD0 = -1.0 #0.2		#if set to -1: randomize; good parameters: 0 to 1



	# parameters for Blazhko effect:
	
	# Blazhko amplitude modulation
	period_blazhko = -1 #10.0		#if set to -1: randomize; good parameters: 10 - 100
	
	# Blazhko frequency modulation
	fac_fm = -1.0 #5.0		# this is the period of the frequency modulation 		#if set to -1: randomize; good parameters: 3.0 to 7.0
	b=-1.0 #-1.0 #3.0			# modulation index describes the factor by which the period is stretched at maximum		#if set to -1: randomize; good parameters: 1.0 to 6.0
		
	if(plot_flag == True):		
		lc_u = np.zeros(len(observation_baseline))
		lc_g = np.zeros(len(observation_baseline))
		lc_r = np.zeros(len(observation_baseline))
		lc_i = np.zeros(len(observation_baseline))
		lc_z = np.zeros(len(observation_baseline))
			
		lc_ampmod_u = np.zeros(len(observation_baseline))
		lc_ampmod_g = np.zeros(len(observation_baseline))
		lc_ampmod_r = np.zeros(len(observation_baseline))
		lc_ampmod_i = np.zeros(len(observation_baseline))
		lc_ampmod_z = np.zeros(len(observation_baseline))	
		
		lc_freqmod_u = np.zeros(len(observation_baseline))
		lc_freqmod_g = np.zeros(len(observation_baseline))
		lc_freqmod_r = np.zeros(len(observation_baseline))
		lc_freqmod_i = np.zeros(len(observation_baseline))
		lc_freqmod_z = np.zeros(len(observation_baseline))

		lc_freqmod_ampmod_u = np.zeros(len(observation_baseline))
		lc_freqmod_ampmod_g = np.zeros(len(observation_baseline))
		lc_freqmod_ampmod_r = np.zeros(len(observation_baseline))
		lc_freqmod_ampmod_i = np.zeros(len(observation_baseline))
		lc_freqmod_ampmod_z = np.zeros(len(observation_baseline))
		
		
	obs_freqmod_u = np.zeros(len(cadence_u))
	obs_freqmod_g = np.zeros(len(cadence_g))
	obs_freqmod_r = np.zeros(len(cadence_r))
	obs_freqmod_i = np.zeros(len(cadence_i))
	obs_freqmod_z = np.zeros(len(cadence_z))
	
	obs_freqmod_ampmod_u = np.zeros(len(cadence_u))
	obs_freqmod_ampmod_g = np.zeros(len(cadence_g))
	obs_freqmod_ampmod_r = np.zeros(len(cadence_r))
	obs_freqmod_ampmod_i = np.zeros(len(cadence_i))
	obs_freqmod_ampmod_z = np.zeros(len(cadence_z))

	obs_freqmod_ampmod_u_with_err = np.zeros(len(cadence_u))
	obs_freqmod_ampmod_g_with_err = np.zeros(len(cadence_g))
	obs_freqmod_ampmod_r_with_err = np.zeros(len(cadence_r))
	obs_freqmod_ampmod_i_with_err = np.zeros(len(cadence_i))
	obs_freqmod_ampmod_z_with_err = np.zeros(len(cadence_z))
	
	
	if(plot_flag == True):
		title_font = {'fontname':'Arial', 'size':'15', 'color':'black', 'weight':'normal',
			'verticalalignment':'bottom'} # Bottom vertical alignment for more space

		axis_font = {'fontname':'Arial', 'size':'11'}

		matplotlib.rc('font', family='sans-serif') 
		###matplotlib.rc('font', serif='Arial') 
		matplotlib.rc('text', usetex='false') 
		matplotlib.rcParams.update({'font.size': 11})
		
		fig = plt.figure(figsize=(11,8))
			
		
	for idx in range(0,true_params.size):
		
	
		period, uA, gA, rA, iA, zA, u0, g0, r0, i0, z0, template_u, template_g, template_r, template_i, template_z = model(idx,true_params)
		
		model_u = uA*template_u + u0
		model_g = gA*template_g + g0 
		model_r = rA*template_r + r0 
		model_i = iA*template_i + i0 
		model_z = zA*template_z + z0
		
		
		# this is an amplitude modulation but with the carrier signal being the RRL template
		
		if(period_blazhko ==-1):
			# randomize
			min_value = 10.0
			max_value = 100.0
			period_blazhko_val =	min_value + (random() * (max_value - min_value))		# draw value between 10 and 100
			
		else:
			period_blazhko_val = period_blazhko
			
			
			
		if(rHJD0 ==-1):
			# randomize
			min_value = 0.0
			max_value = 1.0
			rHJD0_val =	min_value + (random() * (max_value - min_value))		# draw value between 10 and 100
			
		else:
			rHJD0_val = rHJD0

		
		# for testing: only amplitude modulation
		#for i in range(0,len(observation_baseline)):
			
			#t = observation_baseline[i]
			#phase = (t%period)/period + rHJD0_val
			
			#A_blazhko = np.abs(np.sin(t*np.pi/period_blazhko_val))
			##print(A_blazhko)
		
			#if (phase<0):
				#phase=phase+1
			#if (phase>=1):
				#phase=phase-1
			
			#j = np.abs(phase_model - phase).argmin()
			
			## not modulated
			#lc_u[i]=model_u[j]
			#lc_g[i]=model_g[j]
			#lc_r[i]=model_r[j]
			#lc_i[i]=model_i[j]
			#lc_z[i]=model_z[j]
			
			## amplitude modulation
			#lc_ampmod_u[i]=(model_u[j]-np.mean(model_u))*A_blazhko+np.mean(model_u)
			#lc_ampmod_g[i]=(model_g[j]-np.mean(model_g))*A_blazhko+np.mean(model_g)
			#lc_ampmod_r[i]=(model_r[j]-np.mean(model_r))*A_blazhko+np.mean(model_r)
			#lc_ampmod_i[i]=(model_i[j]-np.mean(model_i))*A_blazhko+np.mean(model_i)
			#lc_ampmod_z[i]=(model_z[j]-np.mean(model_z))*A_blazhko+np.mean(model_z)
			
			
	
		# this is a frequency modulation but with the carrier signal being the RRL template
		
		#first do a frequency modulation of a sinusoidal signal and then read off from model_u using the phase
		
		
		
		if(fac_fm ==-1):
			# randomize
			min_value = 3.0
			max_value = 7.0
			fac_fm_val = min_value + (random() * (max_value - min_value))		# draw value between 10 and 100
			
		else:
			fac_fm_val = fac_fm
			
			
		if(b ==-1):
			# randomize
			min_value = 1.0
			max_value = 6.0
			b_val = min_value + (random() * (max_value - min_value))		# draw value between 10 and 100
			
		else:
			b_val = b	
			
			
		fc = 1.0/(period*2*np.pi)   # carrier wave, that's the RR Lyrae lc without frequency modulation
		fm = 1.0/(period*2*np.pi * fac_fm_val) # signal frequency
		
		
		# b = delta f/ fm
		
		#mf = Modulation Index of FM
		#mf = delta f/fm
		#mf is called the modulation index of frequency modulation.

			
		if(plot_flag == True):
			
			current_freq = (2*np.pi*fc*observation_baseline + b*np.sin(2*np.pi*fm*observation_baseline))/observation_baseline
			
			current_period = 1.0/current_freq
			
			phase = (observation_baseline%current_period)/current_period
			
			lc_freqmod_u=[      model_u[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(observation_baseline))    ]
			lc_freqmod_g=[      model_g[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(observation_baseline))    ]
			lc_freqmod_r=[      model_r[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(observation_baseline))    ]
			lc_freqmod_i=[      model_i[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(observation_baseline))    ]
			lc_freqmod_z=[      model_z[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(observation_baseline))    ]

	
			# combine freqmod and ampmod
			
			
			A_blazhko = np.abs(np.sin(observation_baseline*np.pi/period_blazhko_val))
				
			lc_freqmod_ampmod_u = (lc_freqmod_u-np.mean(lc_freqmod_u)) * A_blazhko+np.mean(lc_freqmod_u)
			lc_freqmod_ampmod_g = (lc_freqmod_g-np.mean(lc_freqmod_g)) * A_blazhko+np.mean(lc_freqmod_g)
			lc_freqmod_ampmod_r = (lc_freqmod_r-np.mean(lc_freqmod_r)) * A_blazhko+np.mean(lc_freqmod_r)
			lc_freqmod_ampmod_i = (lc_freqmod_i-np.mean(lc_freqmod_i)) * A_blazhko+np.mean(lc_freqmod_i)
			lc_freqmod_ampmod_z = (lc_freqmod_z-np.mean(lc_freqmod_z)) * A_blazhko+np.mean(lc_freqmod_z)
		
		
		# observations according to LSST cadence
		
				

		#Ivezic, Z., Kahn, S. M., Tyson, A., Abel, b., Acosta, E., et al. 2019, ApJ, 873, 44
		#https://iopscience.iop.org/article/10.3847/1538-4357/ab042c

		#Parameter 	u 	g 	r 	i 	z 	y
		#m_5 	23.78 	24.81 	24.35 	23.92 	23.34 	22.45
		#gamma 0.038 	0.039 	0.039 	0.039 	0.039 	0.039

		# u band
			
		if(len(cadence_u)>0):
			
			current_freq = (2*np.pi*fc*cadence_u + b*np.sin(2*np.pi*fm*cadence_u))/cadence_u
			
			current_period = 1.0/current_freq
	

			phase = (cadence_u%current_period)/current_period
			
			obs_freqmod_u=[      model_u[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(cadence_u))    ]
			
		
			# combine freqmod and ampmod
			
			
			A_blazhko = np.abs(np.sin(cadence_u*np.pi/period_blazhko_val))
				
			obs_freqmod_ampmod_u = (obs_freqmod_u-np.mean(obs_freqmod_u)) * A_blazhko+np.mean(obs_freqmod_u)
		
			
			m_5 =23.78
			gamma = 0.038
			sigma_sys = 0.005
			
		
			X = np.power(10.0,0.4*(obs_freqmod_ampmod_u-m_5))

			sigma_rand = (1/25 - gamma)*X + gamma*np.power(X,2)
				
			sigma_lsst = np.sqrt(np.power(sigma_sys,2) + np.power(sigma_rand,2))

			obs_freqmod_ampmod_u_with_err = obs_freqmod_ampmod_u+ np.random.normal(0.0, sigma_lsst)
			
		
		# g band
		
		if(len(cadence_g)>0):
			
			current_freq = (2*np.pi*fc*cadence_g + b*np.sin(2*np.pi*fm*cadence_g))/cadence_g
			
			current_period = 1.0/current_freq
			
			phase = (cadence_g%current_period)/current_period
			
			obs_freqmod_g=[      model_g[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(cadence_g))    ]
							
			# combine freqmod and ampmod
			
			
			A_blazhko = np.abs(np.sin(cadence_g*np.pi/period_blazhko_val))
				
			obs_freqmod_ampmod_g = (obs_freqmod_g-np.mean(obs_freqmod_g)) * A_blazhko+np.mean(obs_freqmod_g)
			
			
			m_5 = 24.81
			gamma = 0.039
			sigma_sys = 0.005
			
			
			X = np.power(10.0,0.4*(obs_freqmod_ampmod_g-m_5))

			sigma_rand = (1/25 - gamma)*X + gamma*np.power(X,2)
				
			sigma_lsst = np.sqrt(np.power(sigma_sys,2) + np.power(sigma_rand,2))

			obs_freqmod_ampmod_g_with_err = obs_freqmod_ampmod_g+ np.random.normal(0.0, sigma_lsst)
			

		# r band
		
		if(len(cadence_r)>0):
			
			current_freq = (2*np.pi*fc*cadence_r + b*np.sin(2*np.pi*fm*cadence_r))/cadence_r
			
			current_period = 1.0/current_freq
			
			phase = (cadence_r%current_period)/current_period
			
			obs_freqmod_r=[      model_r[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(cadence_r))    ]
			
			A_blazhko = np.abs(np.sin(cadence_r*np.pi/period_blazhko_val))
				
			obs_freqmod_ampmod_r = (obs_freqmod_r-np.mean(obs_freqmod_r)) * A_blazhko+np.mean(obs_freqmod_r)
			
				
			m_5 = 24.35
			gamma = 0.039
			sigma_sys = 0.005	
		
		
			X = np.power(10.0,0.4*(obs_freqmod_ampmod_r-m_5))

			sigma_rand = (1/25 - gamma)*X + gamma*np.power(X,2)
				
			sigma_lsst = np.sqrt(np.power(sigma_sys,2) + np.power(sigma_rand,2))

			obs_freqmod_ampmod_r_with_err = obs_freqmod_ampmod_r+ np.random.normal(0.0, sigma_lsst)
			

		# i band
		
		if(len(cadence_i)>0):
			
			current_freq = (2*np.pi*fc*cadence_i + b*np.sin(2*np.pi*fm*cadence_i))/cadence_i
			
			current_period = 1.0/current_freq
			
			phase = (cadence_i%current_period)/current_period
			
			obs_freqmod_i=[      model_i[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(cadence_i))    ]
					
			
			# combine freqmod and ampmod
			
			A_blazhko = np.abs(np.sin(cadence_i*np.pi/period_blazhko_val))
				
			obs_freqmod_ampmod_i = (obs_freqmod_i-np.mean(obs_freqmod_i)) * A_blazhko+np.mean(obs_freqmod_i)
			
	
	
			m_5 = 23.92
			gamma = 0.039
			sigma_sys = 0.005
			
			
			X = np.power(10.0,0.4*(obs_freqmod_ampmod_i-m_5))

			sigma_rand = (1/25 - gamma)*X + gamma*np.power(X,2)
				
			sigma_lsst = np.sqrt(np.power(sigma_sys,2) + np.power(sigma_rand,2))

			obs_freqmod_ampmod_i_with_err = obs_freqmod_ampmod_i+ np.random.normal(0.0, sigma_lsst)
			
			

		# z band
		
		if(len(cadence_z)>0):
			
			current_freq = (2*np.pi*fc*cadence_z + b*np.sin(2*np.pi*fm*cadence_z))/cadence_z
			
			current_period = 1.0/current_freq
			
			phase = (cadence_z%current_period)/current_period
			
			obs_freqmod_z=[      model_z[np.abs(phase_model - phase[i]).argmin()]  for i in range(0,len(cadence_z))    ]
			
				
			# combine freqmod and ampmod
			
			A_blazhko = np.abs(np.sin(cadence_z*np.pi/period_blazhko_val))
				
			obs_freqmod_ampmod_z = (obs_freqmod_z-np.mean(obs_freqmod_z)) * A_blazhko+np.mean(obs_freqmod_z)
			
			
			m_5 =23.34
			gamma = 0.039
			sigma_sys = 0.005
			
			
			X = np.power(10.0,0.4*(obs_freqmod_ampmod_z-m_5))

			sigma_rand = (1/25 - gamma)*X + gamma*np.power(X,2)
				
			sigma_lsst = np.sqrt(np.power(sigma_sys,2) + np.power(sigma_rand,2))

			obs_freqmod_ampmod_z_with_err = obs_freqmod_ampmod_z+ np.random.normal(0.0, sigma_lsst)
			
			
		print('--------')
		print('idx ', idx)
		print('id ', true_params['id'][idx])
		print('period ', period)
		print('period_blazhko_val ', period_blazhko_val)
		print('rHJD0_val ', rHJD0_val)
		print('fac_fm_val ', fac_fm_val)
		print('b_val ', b_val)

		
		# save light curves
		
		filename_uuid = str(uuid.uuid4())
		f = open("generated_lc/%i_%s.lc"%(idx,filename_uuid), "w")
		f.write('#cadence_u_filename = %s\n' % cadence_u_filename)
		f.write('#cadence_g_filename = %s\n' % cadence_g_filename)
		f.write('#cadence_r_filename = %s\n' % cadence_r_filename)
		f.write('#cadence_i_filename = %s\n' % cadence_i_filename)
		f.write('#cadence_z_filename = %s\n' % cadence_z_filename)
	
		f.write('#idx %i\n'%idx)
		f.write('#id %i\n' % true_params['id'][idx])
		f.write('#period %f\n' % period)
		f.write('#period_blazhko_val %f\n' % period_blazhko_val)
		f.write('#rHJD0_val %f\n' % rHJD0_val)
		f.write('#fac_fm_val %f\n' % fac_fm_val)
		f.write('#b_val %f\n' % b_val)
		
		f.write('#t,mag,filter\n')
		
		
		for k in range(0,len(cadence_u)):
			   
			   f.write('%f,%f,u\n'%(cadence_u[k], obs_freqmod_ampmod_u_with_err[k]))
	
		
		for k in range(0,len(cadence_g)):
			   
			   f.write('%f,%f,g\n'%(cadence_g[k], obs_freqmod_ampmod_g_with_err[k]))	
	
		
		for k in range(0,len(cadence_r)):
			   
			   f.write('%f,%f,r\n'%(cadence_r[k], obs_freqmod_ampmod_r_with_err[k]))	

		
		for k in range(0,len(cadence_i)):
			   
			   f.write('%f,%f,i\n'%(cadence_i[k], obs_freqmod_ampmod_i_with_err[k]))

		
		for k in range(0,len(cadence_z)):
			   
			   f.write('%f,%f,z\n'%(cadence_z[k], obs_freqmod_ampmod_z_with_err[k]))

		f.close()


		if(plot_flag == True):
		
			plt.suptitle('objid=%s'%(true_params['id'][idx]),fontsize=16,x=0.5,y=0.99)   

			gs = gridspec.GridSpec(2, 1) 
					
			ax1 = plt.subplot(gs[0])
			
			ax1.set_xlabel(r"t", **axis_font)
			ax1.set_ylabel(r"mag", **axis_font)
			ax1.invert_yaxis() 
		
			ax1.plot(observation_baseline, lc_freqmod_ampmod_u, 'b-',alpha =0.4, lw=0.01)
			ax1.plot(observation_baseline, lc_freqmod_ampmod_g, 'g-',alpha =0.4, lw=0.01)
			ax1.plot(observation_baseline, lc_freqmod_ampmod_r, 'r-',alpha =0.4, lw=0.01)
			ax1.plot(observation_baseline, lc_freqmod_ampmod_i, 'y-',alpha =0.4, lw=0.01)
			ax1.plot(observation_baseline, lc_freqmod_ampmod_z, 'k-',alpha =0.4, lw=0.01)
		
			ax1.scatter(cadence_u, obs_freqmod_ampmod_u_with_err, color='b',label='u')
			ax1.scatter(cadence_g, obs_freqmod_ampmod_g_with_err, color='g',label='g')
			ax1.scatter(cadence_r, obs_freqmod_ampmod_r_with_err, color='r',label='r')
			ax1.scatter(cadence_i, obs_freqmod_ampmod_i_with_err, color='y',label='i')
			ax1.scatter(cadence_z, obs_freqmod_ampmod_z_with_err, color='k',label='z')
			
			
			ax2 = plt.subplot(gs[1])
			
			ax2.set_xlabel(r"t", **axis_font)
			ax2.set_ylabel(r"mag", **axis_font)
			ax2.invert_yaxis() 
			
			
			length=20
			shorter_baseline = (observation_baseline<observation_baseline[0]+length)
			
			ax2.plot(observation_baseline[shorter_baseline], lc_freqmod_ampmod_u[shorter_baseline], 'b-', alpha =0.4, label='u')
			ax2.plot(observation_baseline[shorter_baseline], lc_freqmod_ampmod_g[shorter_baseline], 'g-', alpha =0.4, label='g')
			ax2.plot(observation_baseline[shorter_baseline], lc_freqmod_ampmod_r[shorter_baseline], 'r-', alpha =0.4, label='r')
			ax2.plot(observation_baseline[shorter_baseline], lc_freqmod_ampmod_i[shorter_baseline], 'y-', alpha =0.4, label='i')
			ax2.plot(observation_baseline[shorter_baseline], lc_freqmod_ampmod_z[shorter_baseline], 'k-', alpha =0.4, label='z')
			
			

			if(len(cadence_u)>0):
				ax2.scatter(cadence_u[cadence_u<np.max(shorter_baseline)], obs_freqmod_ampmod_u_with_err[cadence_u<np.max(shorter_baseline)], color='b',label='u')
				
			if(len(cadence_g)>0):
				ax2.scatter(cadence_g[cadence_g<np.max(shorter_baseline)], obs_freqmod_ampmod_g_with_err[cadence_g<np.max(shorter_baseline)], color='g',label='g')
							
			if(len(cadence_r)>0):
				ax2.scatter(cadence_r[cadence_r<np.max(shorter_baseline)], obs_freqmod_ampmod_r_with_err[cadence_r<np.max(shorter_baseline)], color='r',label='r')
			
			if(len(cadence_i)>0):
				ax2.scatter(cadence_i[cadence_i<np.max(shorter_baseline)], obs_freqmod_ampmod_i_with_err[cadence_i<np.max(shorter_baseline)], color='y',label='i')
				
			if(len(cadence_z)>0):
				ax2.scatter(cadence_z[cadence_z<np.max(shorter_baseline)], obs_freqmod_ampmod_z_with_err[cadence_z<np.max(shorter_baseline)], color='k',label='z')
							
		
			ax1.set_xlim([np.min(observation_baseline)-20,np.max(observation_baseline)+20])
			ax2.set_xlim([np.min(observation_baseline[[shorter_baseline]])-5,np.max(observation_baseline[shorter_baseline])+5])		
			
			ax1.legend(bbox_to_anchor=(1.25, 1.0),  fontsize = 8,title_fontsize=8, title='generated RR Lyrae light curve\nwith parameters:\n \nid %i \nperiod %f \nperiod_blazhko %f \nrHJD0 %f \nfac_fm %f \nb %f'
			%(true_params['id'][idx], period, period_blazhko_val, rHJD0_val, fac_fm_val, b_val))
			
			ax1.set_title("full light curve")
			ax2.set_title("shorter light curve section for illustrative purposes")
			
			# turn off scientific notation on y axis
			plt.ticklabel_format(useOffset=False)
			#plt.subplots_adjust(right=0.7)
			fig.tight_layout()
			
			plt.savefig('_plots/%i_%s_freqmod_ampmod_template_lc.pdf'%(true_params['id'][idx],true_params['type'][idx]))
			
			plt.clf()
	
		

	plt.close()

if __name__ == "__main__":
  main()

