from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
from scipy import interpolate
from scipy.signal import savgol_filter as sgf
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
CC	=	0.73		# Cosmological constant.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.

cm_per_Mpc	=	3.0857 * 1e24


Km_per_Mpc	=	3.0857 * 1e19
s_per_yr	=	365.25 * 24 * 60*60
t_H			=	(1/H_0) * (Km_per_Mpc/s_per_yr)	# Hubble time, in year.


z_min		=	1e-5
z_max		=	3e+1


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.


####################################################################################################################################################






#~ ####################################################################################################################################################
#~ 
#~ 
#~ rho_vs_z__table	=	ascii.read( './../tables/k_correction.txt', format = 'fixed_width' )
#~ z__sim		=	rho_vs_z__table['z'].data
#~ 
#~ 
#~ Reddy_redshiftS			=	np.array([ 2.0, 3.0, 3.8 ])
#~ Reddy_logrhostardotS	=	np.log10( np.array([ 0.152, 0.110, 10**(-1.00) ]) )
#~ Reddy_slope, Reddy_intercept	=	np.polyfit( Reddy_redshiftS, Reddy_logrhostardotS, 1 )
#~ print 'Reddy	:	', Reddy_slope, Reddy_intercept
#~ 
#~ 
#~ Bouwens_redshiftS		=	np.array([ 3.8, 4.9, 5.9, 6.8, 7.9 ])
#~ Bouwens_logrhostardotS	=	np.array([ -1.00, -1.26, -1.55, -1.69, -2.08 ])
#~ Bouwens_slope, Bouwens_intercept	=	np.polyfit( Bouwens_redshiftS, Bouwens_logrhostardotS, 1 )
#~ print 'Bouwens	:	', Bouwens_slope, Bouwens_intercept
#~ 
#~ 
#~ highz_redshiftS			=	np.array([ 7.9, 10.4 ])
#~ highz_logrhostardotS	=	np.array([ -2.08, -3.13 ])
#~ highz_slope, highz_intercept	=	np.polyfit( highz_redshiftS, highz_logrhostardotS, 1 )
#~ print 'highz	:	', highz_slope, highz_intercept
#~ 
#~ 
#~ rho_0	=	10**(-1.7)
#~ norm	=	rho_0 * ( (1+1)**2.0 )
#~ 
#~ rho_star_dot		=	np.ones(z__sim.size)
#~ 
#~ lows				=	np.where( z__sim <= 1.0 )[0]
#~ rho_star_dot[lows]	=	( 1 + z__sim[lows] )**2.5
#~ rho_star_dot[lows]	=	rho_0 * rho_star_dot[lows]
#~ 
#~ inds				=	np.where(  ( 1.0 <= z__sim ) & ( z__sim < 2.0 )  )[0]
#~ rho_star_dot[inds]	=	( 1 + z__sim[inds] )**0.5
#~ rho_star_dot[inds]	=	norm * rho_star_dot[inds]
#~ 
#~ inds				=	np.where(  ( 2.0 <= z__sim ) & ( z__sim < 4.0 )  )[0]
#~ rho_star_dot[inds]	=	10** sf.straight_line( z__sim[inds], Reddy_slope, Reddy_intercept )
#~ 
#~ inds				=	np.where(  ( 4.0 <= z__sim ) & ( z__sim < 7.9 )  )[0]
#~ rho_star_dot[inds]	=	10** sf.straight_line( z__sim[inds], Bouwens_slope, Bouwens_intercept )
#~ 
#~ inds				=	np.where( 7.9 <= z__sim )[0]
#~ rho_star_dot[inds]	=	10** sf.straight_line( z__sim[inds], highz_slope, highz_intercept )
#~ 
#~ 
#~ 
#~ z_min = 0.0 ; z_max = 11.5 ; z_bin = 0.5
#~ y_min = -4.0 ; y_max = 0.0 ; y_bin = 0.2
#~ 
#~ fig	=	plt.figure()
#~ ax	=	fig.add_subplot(111)
#~ ax.set_xlim( z_min, z_max )
#~ ax.set_ylim( y_min, y_max )
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ \rm{ log } \; \dot{ \rho_{\star} } \rm{ ( M_{\odot} yr^{-1} Mpc^{-3} ) } $', fontsize = size_font )
#~ major_ticks = np.arange( z_min, z_max,     2 )
#~ minor_ticks = np.arange( z_min, z_max, z_bin )
#~ ax.set_xticks( major_ticks )                                                       
#~ ax.set_xticks( minor_ticks, minor = True ) 
#~ major_ticks = np.arange( y_min, y_max,     1 )
#~ minor_ticks = np.arange( y_min, y_max, y_bin )
#~ ax.set_yticks( major_ticks )                                                       
#~ ax.set_yticks( minor_ticks, minor = True ) 
#~ ax.plot( Reddy_redshiftS, Reddy_logrhostardotS, 'ko', label = r'$ \rm{ Reddy } $' )
#~ ax.plot( Bouwens_redshiftS, Bouwens_logrhostardotS, 'ro', label = r'$ \rm{ Bouwens } $' )
#~ ax.plot( highz_redshiftS, highz_logrhostardotS, 'go', label = r'$ \rm{ high \, z } $' )
#~ ax.plot( z__sim, np.log10(rho_star_dot), 'b-', label = r'$ \rm{ fit } $' )
#~ plt.legend()
#~ plt.show()
#~ 
#~ comoving_volume	=	np.zeros(z__sim.size)
#~ for j, z in enumerate(z__sim):	
	#~ comoving_volume[j]	=	sf.dVc_by_onepluszee(z)
#~ 
#~ 
#~ rho_vs_z__table	=	Table( [ z__sim, rho_star_dot, comoving_volume ], names = [ 'z', 'rho', 'vol' ] )
#~ ascii.write( rho_vs_z__table, './../tables/rho_star_dot.txt', format = 'fixed_width', overwrite = True )
#~ 
#~ 
#~ 
#~ ####################################################################################################################################################






####################################################################################################################################################


rho_vs_z__table	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )

z__sim		=	rho_vs_z__table['z'].data
CSFR__sim	=	rho_vs_z__table['rho'].data
ind_zMin	=	mf.nearest(z__sim, z_min)
ind_zMax	=	mf.nearest(z__sim, z_max)
z__sim		=	z__sim[   ind_zMin : ind_zMax ]
CSFR__sim	=	CSFR__sim[ind_zMin : ind_zMax ]


####################################################################################################################################################




####################################################################################################################################################



def H_prime(x):
	return np.sqrt(  CC + (1-CC)*( (1+x)**3 )  )


def integrand__t_age(x):
	
	temp	=	(1+x) * H_prime(x)
	return t_H/temp


def f(x, n):
	return x**(-n)


def tau(z, z_prime):
	return	quad( integrand__t_age, z, z_prime )[0]



####################################################################################################################################################




####################################################################################################################################################


#~ n		=	1.0
#~ n		=	1.5
#~ n		=	2.0
#~ n		=	2.5
n		=	3.0

tau_min	=	10*1e6	# lower limit of the delay is set to 10 Myr.

print 'n = ', n, '\n\n'
print 'Hubble time,   in Gyrs		:	', 1e-9*t_H
print 'Minimum delay, in Myrs		:	', 1e-6*tau_min, '\n\n'

####################################################################################################################################################




####################################################################################################################################################


lN	=	int(1e4)

print 'Number of points		:	', z__sim.size
print 'Max z considered		:	', z__sim[:-lN][-1]
print 'Number of points simulated over	:	', z__sim[:-lN].size


####################################################################################################################################################




####################################################################################################################################################



t_age__sim			=	z__sim.copy()
denominator__sim	=	z__sim.copy()
for k, z in enumerate(z__sim):
	
	age					=	quad( integrand__t_age, z, np.inf )[0]
	t_age__sim[k]		=	age
	denominator__sim[k]	=	quad( f, tau_min, age, args=(n) )[0]	



lower_limit_array	=	z__sim.copy()

delay_array	=	t_age__sim - tau_min

for k, z in enumerate(z__sim):
	ind	=	mf.nearest( t_age__sim, delay_array[k] )
	lower_limit_array[k]	=	z__sim[ind]


N	=	int(1e3)
numerator__sim		=	np.zeros(z__sim.size)
t0	=	time.time()
for k, z in enumerate( z__sim[:-lN] ):
	lower_limit			=	lower_limit_array[k]
	
	to_integrate_over	=	np.linspace( lower_limit, z_max, N )
	coefficient_array	=	(1+to_integrate_over) * H_prime(to_integrate_over)
	
	
	ind_min_CSFR		=	mf.nearest( z__sim, lower_limit )
	tck					=	interpolate.splrep( z__sim[ind_min_CSFR:], CSFR__sim[ind_min_CSFR:], s = 5e3 )
	CSFR__interpolated	=	interpolate.splev( to_integrate_over, tck, der = 0 )
	
	coefficient_array	=	(t_H * CSFR__interpolated) / coefficient_array
	
	integrand_array		=	np.zeros(N)
	for j, z_primed in enumerate(to_integrate_over):	integrand_array[j]	=	f( tau(z, z_primed), n )
	integrand_array		=	integrand_array * coefficient_array
			
	numerator__sim[k]	=	simps( integrand_array, to_integrate_over )
print 'Loop done in {:.3f} mins.'.format( ( time.time()-t0 )/60 ), '\n'
CSFR__delayed	=	numerator__sim / denominator__sim
z				=	z__sim[:-lN]
CSFR			=	CSFR__sim[:-lN]
CSFR_delayed	=	CSFR__delayed[:-lN]


Nf = 1000 ; o = 3
CSFR			=	sgf(CSFR, Nf+1, o)
CSFR_delayed	=	sgf(CSFR_delayed, Nf+1, o)

CSFR_delayed_table	=	Table( [ z, CSFR, CSFR_delayed ], names = ['z', 'CSFR', 'CSFR_delayed'] )
ascii.write( CSFR_delayed_table, './../tables/CSFR_delayed--n={0:.1f}.txt'.format(n), format = 'fixed_width', overwrite = True )

