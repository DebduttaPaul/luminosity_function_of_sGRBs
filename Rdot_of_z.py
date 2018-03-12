from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
from scipy.signal import savgol_filter as sgf
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import scipy.io as sio
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


#	n	=	1.0
#	fBC0, in per solar-mass (M-sun)
fBC_min__n1dot0__ECPL	=	9.80*1e-09
fBC_mid__n1dot0__ECPL	=	13.7*1e-09
fBC_max__n1dot0__ECPL	=	14.9*1e-09
fBC_min__n1dot0__BPL	=	2.59*1e-09
fBC_mid__n1dot0__BPL	=	3.74*1e-09
fBC_max__n1dot0__BPL	=	7.50*1e-09

#	n	=	1.5
#	fBC0, in per solar-mass (M-sun)
fBC_min__n1dot5__ECPL	=	5.13*1e-09
fBC_mid__n1dot5__ECPL	=	6.45*1e-09
fBC_max__n1dot5__ECPL	=	6.84*1e-09
fBC_min__n1dot5__BPL	=	1.47*1e-09
fBC_mid__n1dot5__BPL	=	2.05*1e-09
fBC_max__n1dot5__BPL	=	3.78*1e-09

#	n	=	2.0
#	fBC0, in per solar-mass (M-sun)
fBC_min__n2dot0__ECPL	=	3.04*1e-09
fBC_mid__n2dot0__ECPL	=	3.65*1e-09
fBC_max__n2dot0__ECPL	=	3.91*1e-09
fBC_min__n2dot0__BPL	=	0.89*1e-09
fBC_mid__n2dot0__BPL	=	1.23*1e-09
fBC_max__n2dot0__BPL	=	2.17*1e-09


angle_min = 3 ; angle_max = 26	# in degrees
ro	=	3	# digit to which to round off


####################################################################################################################################################




####################################################################################################################################################



k_table		=	ascii.read( './../tables/k_correction.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL			=	k_table['dL'].data

vol_table	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )							;	global volume_term
vol			=	vol_table['vol'].data

ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[: ind_zMax]
dL			=	dL[   : ind_zMax]
vol			=	vol[  : ind_zMax]


LHVKI = sio.loadmat( './../data/LHVKI-detected-BNS-sources.mat' )
LHVKI_dL	= LHVKI['DL'][0]
LHVKI_i		= LHVKI['iota'][0]
LHVKI_SNR	= LHVKI['snrLHVKI'][0]



####################################################################################################################################################






####################################################################################################################################################



n_array	=	np.arange( 1.0, 2.5, 0.5 )
fBC_min__array__ECPL	=	np.array( [fBC_min__n1dot0__ECPL, fBC_min__n1dot5__ECPL, fBC_min__n2dot0__ECPL] )
fBC_mid__array__ECPL	=	np.array( [fBC_mid__n1dot0__ECPL, fBC_mid__n1dot5__ECPL, fBC_mid__n2dot0__ECPL] )
fBC_max__array__ECPL	=	np.array( [fBC_max__n1dot0__ECPL, fBC_max__n1dot5__ECPL, fBC_max__n2dot0__ECPL] )
fBC_min__array__BPL		=	np.array( [fBC_min__n1dot0__BPL , fBC_min__n1dot5__BPL , fBC_min__n2dot0__BPL ] )
fBC_mid__array__BPL		=	np.array( [fBC_mid__n1dot0__BPL , fBC_mid__n1dot5__BPL , fBC_mid__n2dot0__BPL ] )
fBC_max__array__BPL		=	np.array( [fBC_max__n1dot0__BPL , fBC_max__n1dot5__BPL , fBC_max__n2dot0__BPL ] )

ind_zMax	=	mf.nearest( z_sim, z_max )
Rdot_min__array__ECPL	=	np.zeros( (n_array.size, ind_zMax+1) )
Rdot_mid__array__ECPL	=	Rdot_min__array__ECPL.copy()
Rdot_max__array__ECPL	=	Rdot_min__array__ECPL.copy()
Rdot_min__array__BPL	=	np.zeros( (n_array.size, ind_zMax+1) )
Rdot_mid__array__BPL	=	Rdot_min__array__BPL.copy()
Rdot_max__array__BPL	=	Rdot_min__array__BPL.copy()

for g, n in enumerate(n_array):
	
	print 'n:	', n
	
	fBC_min__ECPL	=	fBC_min__array__ECPL[g]
	fBC_mid__ECPL	=	fBC_mid__array__ECPL[g]
	fBC_max__ECPL	=	fBC_max__array__ECPL[g]
	fBC_min__BPL	=	fBC_min__array__BPL[ g]
	fBC_mid__BPL	=	fBC_mid__array__BPL[ g]
	fBC_max__BPL	=	fBC_max__array__BPL[ g]
	
	CSFR_table	=	ascii.read( './../tables/CSFR_delayed--n={0:.1f}.txt'.format(n), format = 'fixed_width' )		;	global Phi
	z			=	CSFR_table['z'].data
	Psi			=	CSFR_table['CSFR_delayed'].data
	
	ind_zMax	=	mf.nearest( z, z_max )
	z			=	z[  :ind_zMax]
	Psi			=	Psi[:ind_zMax]
	#~ print 'Want this to be True:	', ( z - z_sim == 0 ).all()
	#~ print 'Want this to be zero:	', Rdot_min__array__ECPL.shape[1] - (fBC_min__ECPL * Psi * 1e9).size, '\n'	
	
	Rdot_min__array__ECPL[g, :]	=	fBC_min__ECPL * Psi * 1e9
	Rdot_mid__array__ECPL[g, :]	=	fBC_mid__ECPL * Psi * 1e9
	Rdot_max__array__ECPL[g, :]	=	fBC_max__ECPL * Psi * 1e9
	Rdot_min__array__BPL[ g, :]	=	fBC_min__BPL  * Psi * 1e9
	Rdot_mid__array__BPL[ g, :]	=	fBC_mid__BPL  * Psi * 1e9
	Rdot_max__array__BPL[ g, :]	=	fBC_max__BPL  * Psi * 1e9














def Rdotz__Ghirlanda( x, p1, zp, p2 ):
	
	numerator	=	1 + p1*x
	denominator	=	1 + (x/zp)**p2
	
	return numerator / denominator



p1_a = 2.8
zp_a = 2.3
p2_a = 3.5
p1_c = 3.1
zp_c = 2.5
p2_c = 3.6

Ghirlanda_a	=	0.2 * Rdotz__Ghirlanda( z_sim, p1_a, zp_a, p2_a )
Ghirlanda_c	=	0.8 * Rdotz__Ghirlanda( z_sim, p1_c, zp_c, p2_c )

#~ fig	=	plt.figure()
#~ ax	=	fig.add_subplot(111)
#~ ax.set_xlim(0, 6)
#~ 
#~ ax.legend()
#~ plt.show()










Ne = int(1050)

z_min =  0.0 ; z_max = 4.6 ; z_bin = 0.1
y_min = 10**(-1.5) ; y_max = 10**(+2.5)
fig	=	plt.figure()
ax	=	fig.add_subplot(111)
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_min, y_max )
ax.set_yscale('log')
ax.yaxis.set_ticks_position('both')
ax.tick_params( axis = 'y', which = 'both', labelright = 'on' )
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \stackrel{.}{R} \; $' + r'$ \rm{ [yr^{-1} Gpc^{-3}] } $', fontsize = size_font, labelpad = padding-6 )
major_ticks = np.arange( z_min, z_max,    0.5 )
minor_ticks = np.arange( z_min, z_max, z_bin )
ax.set_xticks( major_ticks )
ax.set_xticks( minor_ticks, minor = True )
ax.plot( z, Rdot_mid__array__ECPL[0], linestyle = '--', color = 'C0', linewidth = 3, label = r'$ n = 1.0 $' + r'$ \rm{ , \;   ECPL } $' )
ax.plot( z, Rdot_mid__array__ECPL[1], linestyle = '-' , color = 'C1', linewidth = 3, label = r'$ n = 1.5 $' + r'$ \rm{ , \;   ECPL } $' )
ax.plot( z, Rdot_mid__array__ECPL[2], linestyle = ':' , color = 'C2', linewidth = 3, label = r'$ n = 2.0 $' + r'$ \rm{ , \;   ECPL } $' )
ax.plot( z, Rdot_mid__array__BPL[ 0], linestyle = '-.', color = 'k' , linewidth = 3, label = r'$ n = 1.0 $' + r'$ \rm{ , \; \, BPL } $' )
ax.errorbar( z[Ne], Rdot_mid__array__ECPL[0][Ne], yerr = [  ( Rdot_mid__array__ECPL[0] - Rdot_min__array__ECPL[0] )[Ne]  ] , fmt = '.', ms = marker_size, color = 'C0', markerfacecolor = 'C0', markeredgecolor = 'C0', capsize = 5, zorder = 3 )
ax.errorbar( z[Ne], Rdot_mid__array__ECPL[1][Ne], yerr = [  ( Rdot_mid__array__ECPL[1] - Rdot_min__array__ECPL[1] )[Ne]  ] , fmt = '.', ms = marker_size, color = 'C1', markerfacecolor = 'C1', markeredgecolor = 'C1', capsize = 5, zorder = 3 )
ax.errorbar( z[Ne], Rdot_mid__array__ECPL[2][Ne], yerr = [  ( Rdot_mid__array__ECPL[2] - Rdot_min__array__ECPL[2] )[Ne]  ] , fmt = '.', ms = marker_size, color = 'C2', markerfacecolor = 'C2', markeredgecolor = 'C2', capsize = 5, zorder = 3 )
ax.errorbar( z[Ne], Rdot_mid__array__BPL[ 0][Ne], yerr = [  ( Rdot_mid__array__BPL[ 0] - Rdot_min__array__BPL[ 0] )[Ne]  ] , fmt = '.', ms = marker_size, color = 'k' , markerfacecolor = 'k' , markeredgecolor = 'k' , capsize = 5, zorder = 3 )
ax.plot( z_sim, Ghirlanda_a, linewidth = 1.5, color = 'r', label = r'$ \rm{ Ghirlanda \, (2016) \; model \, [a] } $' )
ax.plot( z_sim, Ghirlanda_c, linewidth = 1.5, color = 'y', label = r'$ \rm{ Ghirlanda \, (2016) \; model \, [c] } $' )
leg	=	ax.legend( loc = 'upper center', bbox_to_anchor = (0.5, 1.00), ncol = 2 )
leg.get_frame().set_edgecolor('k')
plt.savefig( './../plots/Rdot_of_z.png' )
plt.savefig( './../plots/Rdot_of_z.pdf' )
plt.clf()
plt.close()



####################################################################################################################################################
