from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
import debduttaS_functions as mf
import specific_functions as sf
import time, pickle, pprint
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
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
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



####################################################################################################################################################




####################################################################################################################################################


constraints = 4

#~ n	=	1.0
#~ n	=	1.5
n	=	2.0


####################################################################################################################################################




####################################################################################################################################################



k_table		=	ascii.read( './../../tables/k_correction.txt', format = 'fixed_width' )							;	global z_sim, dL_sim, k_Fermi, k_Swift
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_BATSE		=	k_table['k_BATSE'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
ind_zMin	=	mf.nearest(z_sim, z_min)
ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[  ind_zMin : ind_zMax]
dL_sim		=	dL_sim[ ind_zMin : ind_zMax]
k_BATSE		=	k_BATSE[ind_zMin : ind_zMax]
k_Fermi		=	k_Fermi[ind_zMin : ind_zMax]
k_Swift		=	k_Swift[ind_zMin : ind_zMax]

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


Swift_short_Luminosity			=	np.concatenate( [ other_short_Luminosity       , Swift_short_Luminosity       ] )
Swift_short_Luminosity_error	=	np.concatenate( [ other_short_Luminosity_error , Swift_short_Luminosity_error ] )
#	To add artificial errors, of percentage : f
f	=	45.0
Swift_short_Luminosity_error	=	Swift_short_Luminosity_error + (f/100)*Swift_short_Luminosity
N__Swift						=	Swift_short_Luminosity.size
x__Swift_short, y__Swift_short, y__Swift_short_poserr, y__Swift_short_negerr	=	sf.my_histogram_with_errorbars( np.log10(Swift_short_Luminosity/L_norm), np.log10( (Swift_short_Luminosity + Swift_short_Luminosity_error) / L_norm ) - np.log10(Swift_short_Luminosity/L_norm), np.log10( (Swift_short_Luminosity + Swift_short_Luminosity_error) / L_norm ) - np.log10(Swift_short_Luminosity/L_norm), logL_bin*1.0, logL_min, logL_max )
y__Swift_short_error			=	np.maximum(y__Swift_short_negerr, y__Swift_short_poserr)+1
print 'Total number, Swift		:	', N__Swift
print 'Total number, Fermi & Swift	:	', N__Fermi + N__Swift, '\n'


inds_to_delete	=	[]
for j, z in enumerate( BATSE_short_redshift ):
	array	=	np.abs( z_sim - z )
	ind		=	np.where( array == array.min() )[0]
	if ( BATSE_short_Luminosity[j] - L_cut__BATSE[ind] ) < 0 :
		inds_to_delete.append( j )
inds_to_delete	=	np.array( inds_to_delete )
print 'Number of BATSE GRBs		:	', BATSE_short_Luminosity.size
print 'BATSE GRBs, deleted		:	', inds_to_delete.size
BATSE_short_redshift			=	np.delete( BATSE_short_redshift        , inds_to_delete )
BATSE_short_Luminosity			=	np.delete( BATSE_short_Luminosity      , inds_to_delete )
BATSE_short_Luminosity_error	=	np.delete( BATSE_short_Luminosity_error, inds_to_delete )
#	To add artificial errors, of percentage : f
f	=	48.0
BATSE_short_Luminosity_error	=	BATSE_short_Luminosity_error + (f/100)*BATSE_short_Luminosity

N__BATSE						=	BATSE_short_Luminosity.size
x__BATSE_short, y__BATSE_short, y__BATSE_short_poserr, y__BATSE_short_negerr	=	sf.my_histogram_with_errorbars( np.log10(BATSE_short_Luminosity/L_norm), np.log10( (BATSE_short_Luminosity + BATSE_short_Luminosity_error) / L_norm ) - np.log10(BATSE_short_Luminosity/L_norm), np.log10( (BATSE_short_Luminosity + BATSE_short_Luminosity_error) / L_norm ) - np.log10(BATSE_short_Luminosity/L_norm), logL_bin*1.0, logL_min, logL_max )
y__BATSE_short_error			=	np.maximum(y__BATSE_short_negerr, y__BATSE_short_poserr)+1
print '      Number, BATSE		:	', N__BATSE


print '\n'
print 'Fermi error percentage:		', np.mean(Fermi_short_Luminosity_error/Fermi_short_Luminosity)*100
print 'Swift error percentage:		', np.mean(Swift_short_Luminosity_error/Swift_short_Luminosity)*100
print 'BATSE error percentage:		', np.mean(BATSE_short_Luminosity_error/BATSE_short_Luminosity)*100
print '\n'


Luminosity_mids		=	x__Fermi_short
Luminosity_mins		=	L_norm	*	(  10 ** ( Luminosity_mids - logL_bin/2 )  )
Luminosity_maxs		=	L_norm	*	(  10 ** ( Luminosity_mids + logL_bin/2 )  )
L_lo	=	Luminosity_mins.min()
L_hi	=	Luminosity_maxs.max()

print '\n\n'



####################################################################################################################################################





###############################################################################################################################################s



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
		
		integral_over_L[ind_low]	=	(    (  ((Lmax/L_b)[ind_low] )**(Gamma-nu2+1)  )  -  (  ((Lmin/L_b)[ind_low ])**(Gamma-nu2+1)  )    )    /    (Gamma-nu2+1)
		integral_over_L[ind_mid]	=	(    (  1 - ( ((Lmin/L_b)[ind_mid])**(Gamma-nu1+1) )  ) / (Gamma-nu1+1)    )    +    (    (  ( ((Lmax/L_b)[ind_mid])**(Gamma-nu2+1) ) - 1  ) / (Gamma-nu2+1)    )
		integral_over_L[ind_high]	=	(    (  ((Lmax/L_b)[ind_high])**(Gamma-nu1+1)  )  -  (  ((Lmin/L_b)[ind_high])**(Gamma-nu1+1)  )    )    /    (Gamma-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	(  CSFR * den_int/deno  )  *  integral_over_L		
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	norm			=	np.sum(N_vs_L__model)
	
	N_vs_L__model	=	N_vs_L__model / norm
	
	return N_vs_L__model


def find_discrepancy( model, observed ):
	return np.sum(  ( model - observed ) ** 2  )



####################################################################################################################################################






####################################################################################################################################################



print '################################################################################'
print '\n\n'

Gamma__Fermi = 0.001


#~ ##	n = 1.0
#~ Fermi__nu1_array	=	np.array( [0.01, 0.23, 0.48, 0.68, 0.69, 0.70, 0.71] )
#~ Fermi__nu2_array	=	np.array( [1.65, 1.66, 1.67, 1.86, 2.86, 2.93, 2.94, 2.95] )
#~ Fermi__Lb__array	=	np.array( [0.85, 0.86, 0.87, 1.52, 2.92, 3.09, 3.10, 3.11, 3.12] )
#~ ##	n = 1.5
#~ Fermi__nu1_array	=	np.array( [0.01, 0.23, 0.38, 0.58, 0.59, 0.60, 0.61, 0.62] )
#~ Fermi__nu2_array	=	np.array( [1.65, 1.66, 1.67, 1.85, 2.85, 2.87, 2.88, 2.89, 2.90] )
#~ Fermi__Lb__array	=	np.array( [0.84, 0.85, 0.86, 1.46, 2.71, 2.81, 2.82] )
##	n = 2.0
Fermi__nu1_array	=	np.array( [0.01, 0.23, 0.34, 0.55, 0.56, 0.57, 0.58] )
Fermi__nu2_array	=	np.array( [1.65, 1.66, 1.67, 1.85, 2.85, 2.87, 2.88] )
Fermi__Lb__array	=	np.array( [0.85, 0.86, 0.87, 1.45, 2.71, 2.76, 2.77] )

Fermi__nu1_size	=	Fermi__nu1_array.size
Fermi__nu2_size	=	Fermi__nu2_array.size
Fermi__Lb__size	=	Fermi__Lb__array.size
print 'nu1_array:	', Fermi__nu1_array
print 'nu2_array:	', Fermi__nu2_array
print 'Lb__array:	', Fermi__Lb__array, '\n'
grid_of_discrepancy__Fermi	=	np.zeros( (Fermi__nu1_size, Fermi__nu2_size, Fermi__Lb__size ) )
grid_of_rdcdchisqrd__Fermi	=	grid_of_discrepancy__Fermi.copy()
print 'Grid of {0:d} (nu1) X {1:d} (nu2) X {2:d} (Lb) = {3:d}.'.format( Fermi__nu1_size, Fermi__nu2_size, Fermi__Lb__size, grid_of_rdcdchisqrd__Fermi.size), '\n'



t0	=	time.time()
for c1, nu1 in enumerate(Fermi__nu1_array):
	for c2, nu2 in enumerate(Fermi__nu2_array):
		for cLb, coeff in enumerate(Fermi__Lb__array):
					
			model_fit__Fermi	=	model_BPL__Fermi( x__Fermi_short, Gamma__Fermi, nu1, nu2, coeff ) * N__Fermi
			
			grid_of_discrepancy__Fermi[c1, c2, cLb]	=	find_discrepancy( model_fit__Fermi, y__Fermi_short )
			grid_of_rdcdchisqrd__Fermi[c1, c2, cLb]	=	mf.reduced_chisquared( model_fit__Fermi, y__Fermi_short, y__Fermi_short_error, constraints )[2]
print 'Done in {:.3f} seconds.'.format( time.time() - t0 ), '\n\n'

output = open( './../tables/pkl/Fermi--rdcdchisqrd--1.pkl', 'wb' )
pickle.dump( grid_of_rdcdchisqrd__Fermi, output )
output.close()

output = open( './../tables/pkl/Fermi--discrepancy--1.pkl', 'wb' )
pickle.dump( grid_of_discrepancy__Fermi, output )
output.close()


ind_discrepancy_min__Fermi	=	np.unravel_index( grid_of_discrepancy__Fermi.argmin(), grid_of_discrepancy__Fermi.shape )
nu1__Fermi	=	Fermi__nu1_array[ind_discrepancy_min__Fermi[0]]
nu2__Fermi	=	Fermi__nu2_array[ind_discrepancy_min__Fermi[1]]
Lb___Fermi	=	Fermi__Lb__array[ind_discrepancy_min__Fermi[2]]
print 'Minimum discrepancy of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}'.format( grid_of_discrepancy__Fermi[ind_discrepancy_min__Fermi], nu1__Fermi, nu2__Fermi, Lb___Fermi )
print 'Reduced-chisquared of {0:.3f}.'.format( grid_of_rdcdchisqrd__Fermi[ind_discrepancy_min__Fermi]), '\n'

ind_rdcdchisqrd_min__Fermi	=	np.unravel_index( grid_of_rdcdchisqrd__Fermi.argmin(), grid_of_rdcdchisqrd__Fermi.shape )
nu1__Fermi	=	Fermi__nu1_array[ind_rdcdchisqrd_min__Fermi[0]]
nu2__Fermi	=	Fermi__nu2_array[ind_rdcdchisqrd_min__Fermi[1]]
Lb___Fermi	=	Fermi__Lb__array[ind_rdcdchisqrd_min__Fermi[2]]
print 'Minimum reduced-chisquared of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}'.format( grid_of_rdcdchisqrd__Fermi[ind_rdcdchisqrd_min__Fermi], nu1__Fermi, nu2__Fermi, Lb___Fermi )
print 'Reduced-chisquared of {0:.3f}.'.format( grid_of_rdcdchisqrd__Fermi[ind_rdcdchisqrd_min__Fermi]), '\n\n'


grid_of_chisquared__Fermi	=	grid_of_rdcdchisqrd__Fermi * 7

chisquared_at_solution	=	grid_of_chisquared__Fermi[ind_discrepancy_min__Fermi]
chisquared_for_1sigma	=	chisquared_at_solution + 3.53

print 'Chi-squared at 1-sigma:	', np.round(chisquared_for_1sigma, 3), '\n'
print np.round( grid_of_chisquared__Fermi[ :, ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2] ], 3 )
print np.round( grid_of_chisquared__Fermi[ ind_discrepancy_min__Fermi[0], :, ind_discrepancy_min__Fermi[2] ], 3 )
print np.round( grid_of_chisquared__Fermi[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], : ], 3 )


print '\n\n'
print '################################################################################'
