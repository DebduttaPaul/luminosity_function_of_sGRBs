from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
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
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9

logL_bin	=	1.0
logL_min	=	-5.0
logL_max	=	+5.1

z_min		=	1e-1
z_max		=	1e+1


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	10	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.


T_Fermi = 8.9 ; D_Fermi = 1/3
T_ACZTI = 1.0 ; D_ACZTI = 1/3


####################################################################################################################################################




####################################################################################################################################################


constraints = 3

Gamma__Fermi = 0.001 


n	=	1.0
nu__Fermi__min = 0.35 ; Lb__Fermi__min = 5.46
nu__Fermi__mid = 0.71 ; Lb__Fermi__mid = 7.42
nu__Fermi__max = 0.76 ; Lb__Fermi__max = 14.63

#~ n	=	1.5
#~ nu__Fermi__min = 0.25 ; Lb__Fermi__min = 5.26
#~ nu__Fermi__mid = 0.64 ; Lb__Fermi__mid = 6.84
#~ nu__Fermi__max = 0.69 ; Lb__Fermi__max = 13.57

#~ n	=	2.0
#~ nu__Fermi__min = 0.22 ; Lb__Fermi__min = 5.08
#~ nu__Fermi__mid = 0.60 ; Lb__Fermi__mid = 6.61
#~ nu__Fermi__max = 0.65 ; Lb__Fermi__max = 12.70


####################################################################################################################################################




####################################################################################################################################################




k_table		=	ascii.read( './../../tables/k_correction.txt', format = 'fixed_width' )							;	global z_sim, dL_sim, k_Fermi, k_Swift
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_BATSE		=	k_table['k_BATSE'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
k_ACZTI		=	k_table['k_CZTI' ].data
ind_zMin	=	mf.nearest(z_sim, z_min)
ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[  ind_zMin : ind_zMax]
dL_sim		=	dL_sim[ ind_zMin : ind_zMax]
k_BATSE		=	k_BATSE[ind_zMin : ind_zMax]
k_Fermi		=	k_Fermi[ind_zMin : ind_zMax]
k_Swift		=	k_Swift[ind_zMin : ind_zMax]
k_ACZTI		=	k_ACZTI[ind_zMin : ind_zMax]

volume_tab	=	ascii.read( './../../tables/rho_star_dot.txt', format = 'fixed_width' )							;	global volume_term
volume_term	=	volume_tab['vol'].data	;	volume_term	=	volume_term[ind_zMin : ind_zMax]

Phi_table	=	ascii.read( './../../tables/CSFR_delayed--n={0:.1f}.txt'.format(n), format = 'fixed_width' )	;	global Phi
Phi			=	Phi_table['CSFR_delayed'].data	;	Phi			=	Phi[ind_zMin : ind_zMax]

threshold_data	=	ascii.read( './../../tables/thresholds.txt', format = 'fixed_width' )
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data	;	L_cut__Fermi	=	L_cut__Fermi[ind_zMin : ind_zMax]
L_cut__Swift	=	threshold_data['L_cut__Swift'].data	;	L_cut__Swift	=	L_cut__Swift[ind_zMin : ind_zMax]
L_cut__BATSE	=	threshold_data['L_cut__BATSE'].data	;	L_cut__BATSE	=	L_cut__BATSE[ind_zMin : ind_zMax]
L_cut__ACZTI	=	threshold_data['L_cut__CZTI'].data	;	L_cut__ACZTI	=	L_cut__ACZTI[ind_zMin : ind_zMax]



L_vs_z__known_short 	=	ascii.read( './../../tables/L_vs_z__known_short.txt', format = 'fixed_width' )
L_vs_z__Fermi_short 	=	ascii.read( './../../tables/L_vs_z__Fermi_short.txt', format = 'fixed_width' )
L_vs_z__FermE_short 	=	ascii.read( './../../tables/L_vs_z__FermE_short.txt', format = 'fixed_width' )
L_vs_z__Swift_short 	=	ascii.read( './../../tables/L_vs_z__Swift_short.txt', format = 'fixed_width' )
L_vs_z__other_short 	=	ascii.read( './../../tables/L_vs_z__other_short.txt', format = 'fixed_width' )
L_vs_z__BATSE_short 	=	ascii.read( './../../tables/L_vs_z__BATSE_short.txt', format = 'fixed_width' )

known_short_redshift			=	L_vs_z__known_short[ 'measured z'].data
known_short_Luminosity			=	L_vs_z__known_short[ 'Luminosity [erg/s]'].data
known_short_Luminosity_error	=	L_vs_z__known_short[ 'Luminosity_error [erg/s]'].data

Fermi_short_redshift			=	L_vs_z__Fermi_short[  'pseudo z' ].data
Fermi_short_Luminosity			=	L_vs_z__Fermi_short[ 'Luminosity [erg/s]'].data
Fermi_short_Luminosity_error	=	L_vs_z__Fermi_short[ 'Luminosity_error [erg/s]'].data

FermE_short_redshift			=	L_vs_z__FermE_short[ 'pseudo z'].data
FermE_short_Luminosity			=	L_vs_z__FermE_short[ 'Luminosity [erg/s]'].data
FermE_short_Luminosity_error	=	L_vs_z__FermE_short[ 'Luminosity_error [erg/s]'].data

Swift_short_redshift			=	L_vs_z__Swift_short[  'pseudo z' ].data
Swift_short_Luminosity			=	L_vs_z__Swift_short[ 'Luminosity [erg/s]'].data
Swift_short_Luminosity_error	=	L_vs_z__Swift_short[ 'Luminosity_error [erg/s]'].data

other_short_redshift			=	L_vs_z__other_short[ 'measured z'].data
other_short_Luminosity			=	L_vs_z__other_short[ 'Luminosity [erg/s]'].data
other_short_Luminosity_error	=	L_vs_z__other_short[ 'Luminosity_error [erg/s]'].data

BATSE_short_redshift			=	L_vs_z__BATSE_short[ 'pseudo z'].data
BATSE_short_Luminosity			=	L_vs_z__BATSE_short[ 'Luminosity [erg/s]'].data
BATSE_short_Luminosity_error	=	L_vs_z__BATSE_short[ 'Luminosity_error [erg/s]'].data



inds_to_delete	=	np.where(other_short_Luminosity < 1e-16 )[0]
print 'other GRBs, deleted		:	', inds_to_delete.size
other_short_redshift			=	np.delete( other_short_redshift  ,       inds_to_delete )
other_short_Luminosity			=	np.delete( other_short_Luminosity,       inds_to_delete )
other_short_Luminosity_error	=	np.delete( other_short_Luminosity_error, inds_to_delete )

inds_to_delete	=	[]
for j, z in enumerate( Swift_short_redshift ):
	array	=	np.abs( z_sim - z )
	ind		=	np.where( array == array.min() )[0]
	if ( Swift_short_Luminosity[j] - L_cut__Swift[ind] ) < 0 :
		inds_to_delete.append( j )
inds_to_delete	=	np.array( inds_to_delete )
print 'Swift GRBs, deleted		:	', inds_to_delete.size, '\n'
Swift_short_redshift			=	np.delete( Swift_short_redshift        , inds_to_delete )
Swift_short_Luminosity			=	np.delete( Swift_short_Luminosity      , inds_to_delete )
Swift_short_Luminosity_error	=	np.delete( Swift_short_Luminosity_error, inds_to_delete )

inds_to_delete	=	np.where(  Fermi_short_redshift > z_max )[0]
print 'Fermi GRBs, deleted		:	', inds_to_delete.size
Fermi_short_redshift			=	np.delete( Fermi_short_redshift        , inds_to_delete )
Fermi_short_Luminosity			=	np.delete( Fermi_short_Luminosity      , inds_to_delete )
Fermi_short_Luminosity_error	=	np.delete( Fermi_short_Luminosity_error, inds_to_delete )
inds_to_delete	=	np.where(  Swift_short_redshift > z_max )[0]
print 'Swift GRBs, deleted		:	', inds_to_delete.size, '\n'
Swift_short_redshift			=	np.delete( Swift_short_redshift        , inds_to_delete )
Swift_short_Luminosity			=	np.delete( Swift_short_Luminosity      , inds_to_delete )
Swift_short_Luminosity_error	=	np.delete( Swift_short_Luminosity_error, inds_to_delete )

print 'Number of "known" GRBs		:	', known_short_redshift.size
print 'Number of "Fermi" GRBs		:	', Fermi_short_redshift.size
print 'Number of "FermE" GRBs		:	', FermE_short_redshift.size
print 'Number of "Swift" GRBs		:	', Swift_short_redshift.size
print 'Number of "other" GRBs		:	', other_short_redshift.size, '\n'


Fermi_short_Luminosity			=	np.concatenate( [ known_short_Luminosity       , Fermi_short_Luminosity       , FermE_short_Luminosity       ] )
Fermi_short_Luminosity_error	=	np.concatenate( [ known_short_Luminosity_error , Fermi_short_Luminosity_error , FermE_short_Luminosity_error ] )
N__Fermi						=	Fermi_short_Luminosity.size
x__Fermi_short, y__Fermi_short, y__Fermi_short_poserr, y__Fermi_short_negerr	=	sf.my_histogram_with_errorbars( np.log10(Fermi_short_Luminosity/L_norm), np.log10( (Fermi_short_Luminosity + Fermi_short_Luminosity_error) / L_norm ) - np.log10(Fermi_short_Luminosity/L_norm), np.log10( (Fermi_short_Luminosity + Fermi_short_Luminosity_error) / L_norm ) - np.log10(Fermi_short_Luminosity/L_norm), logL_bin*1.0, logL_min, logL_max )
y__Fermi_short_error			=	np.maximum(y__Fermi_short_negerr, y__Fermi_short_poserr)+1
print 'Total number, Fermi		:	', N__Fermi



Luminosity_mids		=	x__Fermi_short
Luminosity_mins		=	L_norm	*	(  10 ** ( Luminosity_mids - logL_bin/2 )  )
Luminosity_maxs		=	L_norm	*	(  10 ** ( Luminosity_mids + logL_bin/2 )  )
L_lo	=	Luminosity_mins.min()
L_hi	=	Luminosity_maxs.max()

print '\n\n'



####################################################################################################################################################





####################################################################################################################################################


def f(x, nu):
	return x**(-nu) * np.exp(-x)


def model_ECPL__Fermi( x__Fermi_short, Gamma, nu, coeff ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * np.ones(z_sim.size)
	
	
	den_int		=	dL_sim**2 * k_Fermi
	den_int		=	den_int ** (-Gamma)
	deno		=	simps( den_int, z_sim )
	
	
	
	lower_limit_array	=	L_lo/L_b
	upper_limit_array	=	L_hi/L_b
	
	denominator			=	np.zeros(z_sim.size)
	for k, z in enumerate(z_sim):
		lower_limit		=	lower_limit_array[k]
		upper_limit		=	upper_limit_array[k]
		
		N	=	1e10	# n = 1.0, 1.5
		#~ N	=	1e9	# n = 2.0
		denominator[k]	=	quad( f, lower_limit, N*lower_limit, args=(-Gamma+nu) )[0]
		
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__Fermi <= L1 )[0]
		Lmin		=	L_cut__Fermi.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		lower_limit_array	=	Lmin/L_b
		upper_limit_array	=	Lmax/L_b
		
		integral_over_L		=	np.zeros(z_sim.size)
		for k, z in enumerate(z_sim):
			lower_limit		=	lower_limit_array[k]
			upper_limit		=	upper_limit_array[k]
		
			integral_over_L[k]	=	quad( f, lower_limit, upper_limit, args=(-Gamma+nu) )[0]
		
		integral_over_L			=	integral_over_L / denominator
		ind						=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand				=	(  CSFR * den_int/deno  )  *  integral_over_L		
		integral				=	simps( integrand, z_sim )
		
		N_vs_L__model[j]		=	integral
	
	
	norm			=	np.sum(N_vs_L__model)
	N_vs_L__model	=	N_vs_L__model / norm
	
	return N_vs_L__model, N__Fermi/norm


def model_ECPL__ACZTI( x__ACZTI_short, Gamma, nu, coeff ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * np.ones(z_sim.size)
	
	
	den_int		=	dL_sim**2 * k_ACZTI
	den_int		=	den_int ** (-Gamma)
	deno		=	simps( den_int, z_sim )
	
	
	
	lower_limit_array	=	L_lo/L_b
	upper_limit_array	=	L_hi/L_b
	
	denominator			=	np.zeros(z_sim.size)
	for k, z in enumerate(z_sim):
		lower_limit		=	lower_limit_array[k]
		upper_limit		=	upper_limit_array[k]
		
		N	=	1e10
		denominator[k]	=	quad( f, lower_limit, N*lower_limit, args=(-Gamma+nu) )[0]
		
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__ACZTI <= L1 )[0]
		Lmin		=	L_cut__ACZTI.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		lower_limit_array	=	Lmin/L_b
		upper_limit_array	=	Lmax/L_b
		
		integral_over_L		=	np.zeros(z_sim.size)
		for k, z in enumerate(z_sim):
			lower_limit		=	lower_limit_array[k]
			upper_limit		=	upper_limit_array[k]
		
			integral_over_L[k]	=	quad( f, lower_limit, upper_limit, args=(-Gamma+nu) )[0]
		
		integral_over_L			=	integral_over_L / denominator
		ind						=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand				=	(  CSFR * den_int/deno  )  *  integral_over_L		
		integral				=	simps( integrand, z_sim )
		
		N_vs_L__model[j]		=	integral
	
	return N_vs_L__model


####################################################################################################################################################






####################################################################################################################################################


nu_array	=	np.array( [nu__Fermi__min, nu__Fermi__mid, nu__Fermi__max] )
Lb_array	=	np.array( [Lb__Fermi__min, Lb__Fermi__mid, Lb__Fermi__max] )

nu_size	=	nu_array.size
Lb_size	=	Lb_array.size
grid_of_norms	=	np.zeros( (nu_size, Lb_size ) )

t0	=	time.time()
for i, nu in enumerate(nu_array):
	for k, Lb in enumerate(Lb_array):
		grid_of_norms[i, k]	=	model_ECPL__Fermi( x__Fermi_short, Gamma__Fermi, nu, Lb )[1]
print 'Done in {:.3f} mins.'.format( (time.time()-t0)/60.0 ), '\n\n'
print 'Grid of norms:	', grid_of_norms, '\n\n'
grid_of_norms	=	grid_of_norms / ( T_Fermi * D_Fermi )
grid_of_norms	=	np.round(grid_of_norms*1e9, 2)


fBC0_min	=	np.min(grid_of_norms)
fBC0_mid	=	grid_of_norms[1,1]
fBC0_max	=	np.max(grid_of_norms)

print '################################################################################'
print '\n'
print 'fBC0: min, mid & max:	', fBC0_min, fBC0_mid, fBC0_max 
print '\n'


ind_norm_min	=	np.unravel_index( grid_of_norms.argmin(), grid_of_norms.shape )
ind_norm_max	=	np.unravel_index( grid_of_norms.argmax(), grid_of_norms.shape )
nu_min			=	nu_array[ind_norm_min[0]]
nu_max			=	nu_array[ind_norm_max[0]]
Lb_min			=	Lb_array[ind_norm_min[1]]
Lb_max			=	Lb_array[ind_norm_max[1]]

print 'arrays...'
print nu_array
print Lb_array, '\n'

print 'min and max at...'
print nu_min, nu_max
print Lb_min, Lb_max

print '\n\n'
print '################################################################################', '\n\n'


####################################################################################################################################################





####################################################################################################################################################




print '################################################################################'
print '\n', 'AstroSat CZTI...', '\n'

norm_ACZTI__Fermi_model__min	=	(fBC0_min*1e-9) * T_ACZTI * D_ACZTI
norm_ACZTI__Fermi_model__mid	=	(fBC0_mid*1e-9) * T_ACZTI * D_ACZTI
norm_ACZTI__Fermi_model__max	=	(fBC0_max*1e-9) * T_ACZTI * D_ACZTI

model_ACZTI__Fermi	=	model_ECPL__ACZTI( x__Fermi_short, Gamma__Fermi, nu__Fermi__mid, Lb__Fermi__mid )

model_ACZTI__Fermi__min			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__min
model_ACZTI__Fermi__mid			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__mid
model_ACZTI__Fermi__max			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__max

print 'A-CZTI rate: min, mid & max	:	', round(np.sum(model_ACZTI__Fermi__min)), round(np.sum(model_ACZTI__Fermi__mid)), round(np.sum(model_ACZTI__Fermi__max)), '\n'
print '################################################################################'



####################################################################################################################################################


