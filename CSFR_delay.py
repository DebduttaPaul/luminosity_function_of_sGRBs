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

#	plt.loglog( z__sim, t_age__sim*1e-9, color = 'k' )
#	plt.ylabel( r'$ t_{\rm{age}} \; \rm{[Gyr]} $', fontsize = size_font )
#~ plt.loglog( z__sim, t_age__sim, color = 'k' )
#~ plt.ylabel( r'$ t_{\rm{age}} \; \rm{[yr]} $', fontsize = size_font )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.show()
#~ 
#~ plt.loglog( z__sim, denominator__sim, color = 'k' )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{denominator} $', fontsize = size_font )
#~ plt.show()




lower_limit_array	=	z__sim.copy()


delay_array	=	t_age__sim - tau_min

#~ plt.loglog( z__sim, t_age__sim, color = 'k' )
#~ plt.ylabel( r'$ t_{\rm{age}} - \tau_{\rm{min}} \; \rm{[yr]} $', fontsize = size_font )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.show()

for k, z in enumerate(z__sim):
	ind	=	mf.nearest( t_age__sim, delay_array[k] )
	lower_limit_array[k]	=	z__sim[ind]

#~ plt.loglog( z__sim, lower_limit_array, color = 'k' )
#~ plt.ylabel( r'$ z_{\rm{min}} $', fontsize = size_font+2 )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.show()
#~ 
#~ 
#~ plt.plot( z__sim, lower_limit_array/z__sim, color = 'k' )
#~ plt.ylabel( r'$ \rm{ratio} $', fontsize = size_font )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.show()




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
	
	#	plt.xlabel( r'$ z $', fontsize = size_font+2 )
	#	plt.loglog( z__sim, CSFR__sim, label = 'full' )
	#	plt.loglog( z__sim[ind_min_CSFR:], CSFR__sim[ind_min_CSFR:], label = 'chosen' )
	#	plt.axvline( x = lower_limit, color = 'k', linestyle = '--' )
	#	plt.axvline( x = z__sim[k], color = 'r', linestyle = '-' )
	#	plt.legend()
	#	plt.show()
	
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
