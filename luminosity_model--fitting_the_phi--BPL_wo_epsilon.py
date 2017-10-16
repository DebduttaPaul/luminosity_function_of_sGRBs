from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.stats import pearsonr as R
from scipy.stats import spearmanr as S
from scipy.stats import kendalltau as T
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
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

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9

logL_bin	=	1.0
logL_min	=	-5
logL_max	=	+5

z_min		=	1e-1
z_max		=	1e+1


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



####################################################################################################################################################






####################################################################################################################################################




k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )				;	global z_sim
z_sim		=	k_table['z'].data
ind_zMin	=	mf.nearest(z_sim, z_min)
ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[ind_zMin : ind_zMax]

volume_tab	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )		;	global volume_term
volume_term	=	volume_tab['vol'].data	;	volume_term	=	volume_term[ind_zMin : ind_zMax]

Phi_table	=	ascii.read( './../tables/CSFR_delayed--n=1.0.txt', format = 'fixed_width' )	;	global Phi
Phi			=	Phi_table['CSFR_delayed'].data	;	Phi			=	Phi[ind_zMin : ind_zMax]

threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data	;	L_cut__Fermi	=	L_cut__Fermi[ind_zMin : ind_zMax]
L_cut__Swift	=	threshold_data['L_cut__Swift'].data	;	L_cut__Swift	=	L_cut__Swift[ind_zMin : ind_zMax]
L_cut__BATSE	=	threshold_data['L_cut__BATSE'].data	;	L_cut__BATSE	=	L_cut__BATSE[ind_zMin : ind_zMax]
L_cut__CZTI		=	threshold_data['L_cut__CZTI'].data	;	L_cut__CZTI		=	L_cut__CZTI[ ind_zMin : ind_zMax]



L_vs_z__known_short 	=	ascii.read( './../tables/L_vs_z__known_short.txt', format = 'fixed_width' )
L_vs_z__Fermi_short 	=	ascii.read( './../tables/L_vs_z__Fermi_short.txt', format = 'fixed_width' )
L_vs_z__FermE_short 	=	ascii.read( './../tables/L_vs_z__FermE_short.txt', format = 'fixed_width' )
L_vs_z__Swift_short 	=	ascii.read( './../tables/L_vs_z__Swift_short.txt', format = 'fixed_width' )
L_vs_z__other_short 	=	ascii.read( './../tables/L_vs_z__other_short.txt', format = 'fixed_width' )
L_vs_z__BATSE_short 	=	ascii.read( './../tables/L_vs_z__BATSE_short.txt', format = 'fixed_width' )

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
f	=	50.0
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
f	=	50.0
BATSE_short_Luminosity_error	=	BATSE_short_Luminosity_error + (f/100)*BATSE_short_Luminosity

N__BATSE						=	BATSE_short_Luminosity.size
x__BATSE_short, y__BATSE_short, y__BATSE_short_poserr, y__BATSE_short_negerr	=	sf.my_histogram_with_errorbars( np.log10(BATSE_short_Luminosity/L_norm), np.log10( (BATSE_short_Luminosity + BATSE_short_Luminosity_error) / L_norm ) - np.log10(BATSE_short_Luminosity/L_norm), np.log10( (BATSE_short_Luminosity + BATSE_short_Luminosity_error) / L_norm ) - np.log10(BATSE_short_Luminosity/L_norm), logL_bin*1.0, logL_min, logL_max )
y__BATSE_short_error			=	np.maximum(y__BATSE_short_negerr, y__BATSE_short_poserr)+1
print '      Number, BATSE		:	', N__BATSE




Luminosity_mids		=	x__Fermi_short
Luminosity_mins		=	L_norm	*	(  10 ** ( Luminosity_mids - logL_bin/2 )  )
Luminosity_maxs		=	L_norm	*	(  10 ** ( Luminosity_mids + logL_bin/2 )  )
L_lo	=	Luminosity_mins.min()
L_hi	=	Luminosity_maxs.max()

print '\n\n'



####################################################################################################################################################





####################################################################################################################################################



def model_BPL__Fermi( x__Fermi_short, nu1, nu2, coeff, delta ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * ( (1+z_sim)**delta )
	denominator	=	(  ( 1 - (L_lo/L_b)**(-nu1+1) ) / (-nu1+1)  )  +  (  ( (L_hi/L_b)**(-nu2+1) - 1 ) / (-nu2+1)  )
	
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
		
		integral_over_L[ind_low]	=	(  ((Lmax/L_b)[ind_low])**(-nu2+1) - ((Lmin/L_b)[ind_low])**(-nu2+1)  ) / (-nu2+1)
		integral_over_L[ind_mid]	=	(  ( 1 - ((Lmin/L_b)[ind_mid])**(-nu1+1) ) / (-nu1+1)  )  +  (  ( ((Lmax/L_b)[ind_mid])**(-nu2+1) - 1 ) / (-nu2+1)  )
		integral_over_L[ind_high]	=	(  ((Lmax/L_b)[ind_high])**(-nu1+1) - ((Lmin/L_b)[ind_high])**(-nu1+1)  ) / (-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	CSFR  *  integral_over_L
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	norm			=	np.sum(N_vs_L__model)
	
	N_vs_L__model	=	N_vs_L__model / norm
	
	
	return N_vs_L__model


def model_BPL__Swift( x__Swift_short, nu1, nu2, coeff, delta ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * ( (1+z_sim)**delta )
	denominator	=	(  ( 1 - (L_lo/L_b)**(-nu1+1) ) / (-nu1+1)  )  +  (  ( (L_hi/L_b)**(-nu2+1) - 1 ) / (-nu2+1)  )
	
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__Swift <= L1 )[0]
		Lmin		=	L_cut__Swift.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		ind_low			=	np.where( L_b <= L1 )[0]
		ind_mid			=	np.where( (L1 < L_b) & (L_b < L2) )[0]
		ind_high		=	np.where( L2 <= L_b )[0]
		
		integral_over_L[ind_low]	=	(  ((Lmax/L_b)[ind_low])**(-nu2+1) - ((Lmin/L_b)[ind_low])**(-nu2+1)  ) / (-nu2+1)
		integral_over_L[ind_mid]	=	(  ( 1 - ((Lmin/L_b)[ind_mid])**(-nu1+1) ) / (-nu1+1)  )  +  (  ( ((Lmax/L_b)[ind_mid])**(-nu2+1) - 1 ) / (-nu2+1)  )
		integral_over_L[ind_high]	=	(  ((Lmax/L_b)[ind_high])**(-nu1+1) - ((Lmin/L_b)[ind_high])**(-nu1+1)  ) / (-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	CSFR  *  integral_over_L
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	norm			=	np.sum(N_vs_L__model)
	
	N_vs_L__model	=	N_vs_L__model / norm
	
	return N_vs_L__model


def model_BPL__BATSE( x__BATSE_short, nu1, nu2, coeff, delta ):
	
	
	CSFR		=	Phi * volume_term
	
	L_b			=	( L_norm * coeff ) * ( (1+z_sim)**delta )
	denominator	=	(  ( 1 - (L_lo/L_b)**(-nu1+1) ) / (-nu1+1)  )  +  (  ( (L_hi/L_b)**(-nu2+1) - 1 ) / (-nu2+1)  )
	
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut__BATSE <= L1 )[0]
		Lmin		=	L_cut__BATSE.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		ind_low			=	np.where( L_b <= L1 )[0]
		ind_mid			=	np.where( (L1 < L_b) & (L_b < L2) )[0]
		ind_high		=	np.where( L2 <= L_b )[0]
		
		integral_over_L[ind_low]	=	(  ((Lmax/L_b)[ind_low])**(-nu2+1) - ((Lmin/L_b)[ind_low])**(-nu2+1)  ) / (-nu2+1)
		integral_over_L[ind_mid]	=	(  ( 1 - ((Lmin/L_b)[ind_mid])**(-nu1+1) ) / (-nu1+1)  )  +  (  ( ((Lmax/L_b)[ind_mid])**(-nu2+1) - 1 ) / (-nu2+1)  )
		integral_over_L[ind_high]	=	(  ((Lmax/L_b)[ind_high])**(-nu1+1) - ((Lmin/L_b)[ind_high])**(-nu1+1)  ) / (-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		integrand			=	CSFR  *  integral_over_L		
		N_vs_L__model[j]	=	simps( integrand, z_sim )
	
	norm			=	np.sum(N_vs_L__model)
	
	N_vs_L__model	=	N_vs_L__model / norm
	
	return N_vs_L__model



####################################################################################################################################################






####################################################################################################################################################



nu1_l = 0.10 ; nu2_l = 3.10 ; coeff_l = 0.30 ; delta_l = 2.90

lower_bounds	=	[0.0, 0.0, 0.0, 0.0]
upper_bounds	=	[4.0, 8.0, 8.0, 5.0]




print '################################################################################'
print '\n', 'Fermi...', '\n\n'

popt__Fermi, pcov__Fermi	=	curve_fit( model_BPL__Fermi, x__Fermi_short, y__Fermi_short/N__Fermi, p0 = [nu1_l, nu2_l, coeff_l, delta_l], bounds = (lower_bounds, upper_bounds)  )
model_fit__Fermi			=	model_BPL__Fermi( x__Fermi_short, *popt__Fermi ) * N__Fermi
nu1__Fermi	=	popt__Fermi[0]
nu2__Fermi	=	popt__Fermi[1]
coeff__Fermi=	popt__Fermi[2]
delta__Fermi=	popt__Fermi[3]
print 'nu1__Fermi	:	', round(nu1__Fermi   , 3)
print 'nu2__Fermi	:	', round(nu2__Fermi   , 3)
print 'coeff__Fermi	:	', round(coeff__Fermi , 3)
print 'delta__Fermi	:	', round(delta__Fermi , 3)
print '\n', np.round(pcov__Fermi, 3)
print '\n', 'red_chisquared	:	', mf.reduced_chisquared( model_fit__Fermi, y__Fermi_short, y__Fermi_short_error, 5 ), '\n\n'

plt.title( r'$ Fermi $' )
ax	=	plt.subplot(111)
ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
ax.errorbar( x__Fermi_short, y__Fermi_short, yerr = [ y__Fermi_short_poserr, y__Fermi_short_negerr ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
ax.plot( x__Fermi_short, model_fit__Fermi, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
ax.legend()
plt.savefig('./../plots/estimated_lumfunc_models/Fermi--delta.png')
plt.clf()
plt.close()


print '################################################################################'



print '\n\n\n\n'



print '################################################################################'
print '\n', 'Swift...', '\n\n'

popt__Swift, pcov__Swift	=	curve_fit( model_BPL__Swift, x__Swift_short, y__Swift_short/N__Swift, p0 = [nu1_l, nu2_l, coeff_l, delta_l], bounds = (lower_bounds, upper_bounds)  )
model_fit__Swift			=	model_BPL__Swift( x__Swift_short, *popt__Swift )  * N__Swift
nu1__Swift	=	popt__Swift[0]
nu2__Swift	=	popt__Swift[1]
coeff__Swift=	popt__Swift[2]
delta__Swift=	popt__Swift[3]
print 'nu1__Swift	:	', round(nu1__Swift   , 3)
print 'nu2__Swift	:	', round(nu2__Swift   , 3)
print 'coeff__Swift	:	', round(coeff__Swift , 3)
print 'delta__Swift	:	', round(delta__Swift , 3)
print '\n', np.round(pcov__Swift, 3)
print '\n', 'red_chisquared	:	', mf.reduced_chisquared( model_fit__Swift, y__Swift_short, y__Swift_short_error, 5 ), '\n\n'

plt.title( r'$ Swift $' )
ax	=	plt.subplot(111)
ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
ax.errorbar( x__Swift_short, y__Swift_short, yerr = [ y__Swift_short_poserr, y__Swift_short_negerr ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
ax.plot( x__Swift_short, model_fit__Swift, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
ax.legend()
plt.savefig('./../plots/estimated_lumfunc_models/Swift--delta.png')
plt.clf()
plt.close()


print '################################################################################'



print '\n\n\n\n'



print '################################################################################'
print '\n', 'BATSE...', '\n\n'

popt__BATSE, pcov__BATSE	=	curve_fit( model_BPL__BATSE, x__BATSE_short, y__BATSE_short/N__BATSE, p0 = [nu1_l, nu2_l, coeff_l, delta_l], bounds = (lower_bounds, upper_bounds)  )
model_fit__BATSE			=	model_BPL__BATSE( x__BATSE_short, *popt__BATSE )  * N__BATSE
nu1__BATSE	=	popt__BATSE[0]
nu2__BATSE	=	popt__BATSE[1]
coeff__BATSE=	popt__BATSE[2]
delta__BATSE=	popt__BATSE[3]
print 'nu1__BATSE	:	', round(nu1__BATSE  , 3)
print 'nu2__BATSE	:	', round(nu2__BATSE  , 3)
print 'coeff__BATSE	:	', round(coeff__BATSE, 3)
print 'delta__BATSE	:	', round(delta__BATSE, 3)
print '\n', np.round(pcov__BATSE, 3)
print '\n', 'red_chisquared	:	', mf.reduced_chisquared( model_fit__BATSE, y__BATSE_short, y__BATSE_short_error, 5 ), '\n\n'

plt.title( r'$ BATSE $' )
ax	=	plt.subplot(111)
ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
ax.errorbar( x__BATSE_short, y__BATSE_short, yerr = [ y__BATSE_short_poserr, y__BATSE_short_negerr ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
ax.plot( x__BATSE_short, model_fit__BATSE, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
ax.legend()
plt.savefig('./../plots/estimated_lumfunc_models/BATSE--delta.png')
plt.clf()
plt.close()


print '################################################################################'




