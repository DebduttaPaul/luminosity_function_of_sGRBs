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


constraints = 4


#~ n	=	1.0
#~ n	=	1.5
n	=	2.0


Gamma__Fermi = 0.001 


#~ #	n = 1.0
#~ nu1__Fermi__min = 0.01 ; nu2__Fermi__min = 1.66 ; Lb0__Fermi__min = 0.85
#~ nu1__Fermi__mid = 0.48 ; nu2__Fermi__mid = 1.86 ; Lb0__Fermi__mid = 1.52
#~ nu1__Fermi__max = 0.70 ; nu2__Fermi__max = 2.94 ; Lb0__Fermi__max = 3.10
#~ #	n = 1.5
#~ nu1__Fermi__min = 0.01 ; nu2__Fermi__min = 1.66 ; Lb0__Fermi__min = 0.84
#~ nu1__Fermi__mid = 0.38 ; nu2__Fermi__mid = 1.85 ; Lb0__Fermi__mid = 1.46
#~ nu1__Fermi__max = 0.61 ; nu2__Fermi__max = 2.89 ; Lb0__Fermi__max = 2.82
#	n = 2.0
nu1__Fermi__min = 0.01 ; nu2__Fermi__min = 1.66 ; Lb0__Fermi__min = 0.85
nu1__Fermi__mid = 0.34 ; nu2__Fermi__mid = 1.85 ; Lb0__Fermi__mid = 1.45
nu1__Fermi__max = 0.57 ; nu2__Fermi__max = 2.88 ; Lb0__Fermi__max = 2.77


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


def model_BPL__Fermi( x__Fermi_short, Gamma, nu1, nu2, coeff ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * np.ones(z_sim.size)
	
	
	den_int		=	dL_sim**2 * k_Fermi
	den_int		=	den_int ** (-Gamma)
	deno		=	simps( den_int, z_sim )
	
	
	denominator	=	(  ( 1 - ((L_lo/L_b)**(Gamma-nu1+1)) ) / (Gamma-nu1+1)  )  +  (  ( ((L_hi/L_b)**(Gamma-nu2+1)) - 1 ) / (Gamma-nu2+1)  )
	
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__Fermi <= L1 )[0]
		Lmin		=	L_cut__Fermi.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		ind_low			=	np.where( L_b <= L1 )[0]
		ind_mid			=	np.where( (L1 < L_b) & (L_b < L2) )[0]
		ind_high		=	np.where( L2 <= L_b )[0]
		
		integral_over_L[ind_low]	=	(    (  ((Lmax/L_b)[ind_low ])**(Gamma-nu2+1)  )  -  (  ((Lmin/L_b)[ind_low ])**(Gamma-nu2+1)  )    )    /    (Gamma-nu2+1)
		integral_over_L[ind_mid]	=	(    (  1 - ( ((Lmin/L_b)[ind_mid])**(Gamma-nu1+1) )  ) / (Gamma-nu1+1)    )    +    (    (  ( ((Lmax/L_b)[ind_mid])**(Gamma-nu2+1) ) - 1  ) / (Gamma-nu2+1)    )
		integral_over_L[ind_high]	=	(    (  ((Lmax/L_b)[ind_high])**(Gamma-nu1+1)  )  -  (  ((Lmin/L_b)[ind_high])**(Gamma-nu1+1)  )    )    /    (Gamma-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	(  CSFR * den_int/deno  )  *  integral_over_L		
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	norm			=	np.sum(N_vs_L__model)
	N_vs_L__model	=	N_vs_L__model / norm
	
	return N_vs_L__model, N__Fermi/norm


def model_BPL__ACZTI( x__ACZTI_short, Gamma, nu1, nu2, coeff ):
		
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * np.ones(z_sim.size)
	
	
	den_int		=	dL_sim**2 * k_ACZTI
	den_int		=	den_int ** (-Gamma)
	deno		=	simps( den_int, z_sim )
	
	
	denominator	=	(  ( 1 - ((L_lo/L_b)**(Gamma-nu1+1)) ) / (Gamma-nu1+1)  )  +  (  ( ((L_hi/L_b)**(Gamma-nu2+1)) - 1 ) / (Gamma-nu2+1)  )
	
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__ACZTI <= L1 )[0]
		Lmin		=	L_cut__ACZTI.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		ind_low			=	np.where( L_b <= L1 )[0]
		ind_mid			=	np.where( (L1 < L_b) & (L_b < L2) )[0]
		ind_high		=	np.where( L2 <= L_b )[0]
		
		integral_over_L[ind_low]	=	(    (  ((Lmax/L_b)[ind_low ])**(Gamma-nu2+1)  )  -  (  ((Lmin/L_b)[ind_low ])**(Gamma-nu2+1)  )    )    /    (Gamma-nu2+1)
		integral_over_L[ind_mid]	=	(    (  1 - ( ((Lmin/L_b)[ind_mid])**(Gamma-nu1+1) )  ) / (Gamma-nu1+1)    )    +    (    (  ( ((Lmax/L_b)[ind_mid])**(Gamma-nu2+1) ) - 1  ) / (Gamma-nu2+1)    )
		integral_over_L[ind_high]	=	(    (  ((Lmax/L_b)[ind_high])**(Gamma-nu1+1)  )  -  (  ((Lmin/L_b)[ind_high])**(Gamma-nu1+1)  )    )    /    (Gamma-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	(  CSFR * den_int/deno  )  *  integral_over_L		
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	return N_vs_L__model


####################################################################################################################################################






####################################################################################################################################################


nu1_array	=	np.array( [nu1__Fermi__min, nu1__Fermi__mid, nu1__Fermi__max] )
nu2_array	=	np.array( [nu2__Fermi__min, nu2__Fermi__mid, nu2__Fermi__max] )
Lb0_array	=	np.array( [Lb0__Fermi__min, Lb0__Fermi__mid, Lb0__Fermi__max] )


nu1_size	=	nu1_array.size
nu2_size	=	nu2_array.size
Lb0_size	=	Lb0_array.size
grid_of_norm	=	np.zeros( (nu1_size, nu2_size, Lb0_size ) )

for i, nu1 in enumerate(nu1_array):
	for j, nu2 in enumerate(nu2_array):
		for k, Lb0 in enumerate(Lb0_array):
			grid_of_norm[i, j, k]	=	model_BPL__Fermi( x__Fermi_short, Gamma__Fermi, nu1, nu2, Lb0 )[1]
grid_of_norm	=	grid_of_norm / ( T_Fermi * D_Fermi )
grid_of_norm	=	np.round(grid_of_norm*1e9, 2)



fBC0_min	=	np.min(grid_of_norm)
fBC0_mid	=	grid_of_norm[1,1,1]
fBC0_max	=	np.max(grid_of_norm)

print '################################################################################'
print '\n'
print 'fBC0: min, mid & max:	', fBC0_min, fBC0_mid, fBC0_max 
print '\n'


ind_norm_min	=	np.unravel_index( grid_of_norm.argmin(), grid_of_norm.shape )
ind_norm_max	=	np.unravel_index( grid_of_norm.argmax(), grid_of_norm.shape )
nu1_min			=	nu1_array[ind_norm_min[0]]
nu1_max			=	nu1_array[ind_norm_max[0]]
nu2_min			=	nu2_array[ind_norm_min[1]]
nu2_max			=	nu2_array[ind_norm_max[1]]
Lb0_min			=	Lb0_array[ind_norm_min[2]]
Lb0_max			=	Lb0_array[ind_norm_max[2]]

print 'arrays...'
print nu1_array
print nu2_array
print Lb0_array, '\n'

print 'min and max at...'
print nu1_min, nu1_max
print nu2_min, nu2_max
print Lb0_min, Lb0_max

print '\n\n'
print '################################################################################', '\n\n'


####################################################################################################################################################





####################################################################################################################################################




print '################################################################################'
print '\n', 'AstroSat CZTI...', '\n'

norm_ACZTI__Fermi_model__min	=	(fBC0_min*1e-9) * T_ACZTI * D_ACZTI
norm_ACZTI__Fermi_model__mid	=	(fBC0_mid*1e-9) * T_ACZTI * D_ACZTI
norm_ACZTI__Fermi_model__max	=	(fBC0_max*1e-9) * T_ACZTI * D_ACZTI

model_ACZTI__Fermi	=	model_BPL__ACZTI( x__Fermi_short, Gamma__Fermi, nu1__Fermi__mid, nu2__Fermi__mid, Lb0__Fermi__mid )

model_ACZTI__Fermi__min			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__min
model_ACZTI__Fermi__mid			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__mid
model_ACZTI__Fermi__max			=	model_ACZTI__Fermi * norm_ACZTI__Fermi_model__max

print 'A-CZTI rate: min, mid & max	:	', round(np.sum(model_ACZTI__Fermi__min)), round(np.sum(model_ACZTI__Fermi__mid)), round(np.sum(model_ACZTI__Fermi__max)), '\n'
print '################################################################################'



####################################################################################################################################################


