from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
from scipy import interpolate
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10', size = 12)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.


z_max		=	1e+1


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.


####################################################################################################################################################




####################################################################################################################################################



CSFR_table	=	ascii.read( './../tables/CSFR_delayed--n=1.0.txt', format = 'fixed_width' )
z			=	CSFR_table['z'].data
Psi			=	CSFR_table['CSFR_delayed'].data

k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL			=	k_table['dL'].data

vol_table	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )
vol			=	vol_table['vol'].data



####################################################################################################################################################




####################################################################################################################################################





####################################################################################################################################################



def correct_theta( rate, theta ):
	
	beaming_factor	=	1 - np.cos( theta*(P/180) )
	return rate / beaming_factor


####################################################################################################################################################




ind_zMax	=	mf.nearest( z   , z_max)
z			=	z[  :ind_zMax]
Psi			=	Psi[:ind_zMax]

ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[: ind_zMax]
dL			=	dL[   : ind_zMax]
vol			=	vol[  : ind_zMax]

#	print ( z - z_sim == 0 ).all()





ind	= mf.nearest( dL, 201 )

z			=	z[  : ind]
dL			=	dL[ : ind]
Psi			=	Psi[: ind]
vol			=	vol[: ind]

print 'z_max for dL =', dL[-1], '[Mpc] is ', z[-1], '.'
#	plt.loglog( z, dL )
#	plt.show()




fBC_min	=	1.1246048473694117e-09
fBC_max	=	2.3140623675977014e-09


V	=	(4*P/3)*(0.2**3)
#~ print '\n'
#~ print 'total volume [Mpc^3] : ', V

R0_min	=	fBC_min * Psi[0] * 1e9
R0_max	=	fBC_max * Psi[0] * 1e9
print '\n'
print 'R0, min [ yr^{-1} Gpc^{-3} ] : ', R0_min
print 'R0, max [ yr^{-1} Gpc^{-3} ] : ', R0_max

print '\n'
print 'Correct for beaming, with '
print correct_theta( R0_min, 26 )
#~ print correct_theta( R0_max, 26 )
#~ print correct_theta( R0_min,  3 )
print correct_theta( R0_max,  3 )





integrated_rate_min__without_correction	=	(4*P) * simps( fBC_min*Psi*vol, z  )
integrated_rate_max__without_correction	=	(4*P) * simps( fBC_max*Psi*vol, z  )
print '\n\n\n'
print 'Integrated rates, without correction'
print integrated_rate_min__without_correction
print integrated_rate_max__without_correction

print '\n'
print 'With correction'
print correct_theta( integrated_rate_min__without_correction, 26 )
#~ print correct_theta( integrated_rate_max__without_correction, 26 )
#~ print correct_theta( integrated_rate_min__without_correction,  3 )
print correct_theta( integrated_rate_max__without_correction,  3 )
