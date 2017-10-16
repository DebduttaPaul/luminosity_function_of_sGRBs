from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.stats import pearsonr as R
from scipy.stats import spearmanr as S
from scipy.stats import kendalltau as T
from scipy.optimize import curve_fit
from scipy.integrate import quad
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10', size = 15)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9

padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	2e+1 #	for the purposes of plotting
x_in_keV_min	=	1e00	;	x_in_keV_max	=	2e04	#	Ep(1+z), min & max.
y_in_eps_min	=	1e49	;	y_in_eps_max	=	1e56	#	L_iso  , min & max.


####################################################################################################################################################




####################################################################################################################################################


def straight_line( x, m, c ):
	return m*x + c


def fit_the_trend( x, y ):
	
	r, p_r	=	R( x, y )
	s, p_s	=	S( x, y )
	t, p_t	=	T( x, y )
	print 'R, p			:	', r, p_r
	print 'S, p			:	', s, p_s
	print 'T, p			:	', t, p_t
	
	x_to_fit_log	=	np.log10( x )
	y_to_fit_log	=	np.log10( y )
	
	popt, pcov		=	curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
	exponent		=	popt[0]		;	coefficient			=	10**popt[1] *L_norm
	exponent_error	=	pcov[0,0]	;	coefficient_error	=	( 10**pcov[1,1] - 1 ) * coefficient
	print 'Coefficient		:	',	round(coefficient, 3), round(coefficient_error, 5)
	print 'Exponent		:	',		round(   exponent, 3), round(   exponent_error, 3)
	print '\n\n\n\n'
	
	return coefficient, coefficient_error, exponent, exponent_error


####################################################################################################################################################




####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__BATSE	=	threshold_data['L_cut__BATSE'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data
L_cut__CZTI		=	threshold_data['L_cut__CZTI' ].data


L_vs_z__known_short	=	ascii.read( './../tables/L_vs_z__known_short.txt', format = 'fixed_width' )
L_vs_z__Fermi_short	=	ascii.read( './../tables/L_vs_z__Fermi_short.txt', format = 'fixed_width' )
L_vs_z__Swift_short	=	ascii.read( './../tables/L_vs_z__Swift_short.txt', format = 'fixed_width' )
L_vs_z__other_short	=	ascii.read( './../tables/L_vs_z__other_short.txt', format = 'fixed_width' )
L_vs_z__BATSE_short	=	ascii.read( './../tables/L_vs_z__BATSE_short.txt', format = 'fixed_width' )
L_vs_z__FermE_short	=	ascii.read( './../tables/L_vs_z__FermE_short.txt', format = 'fixed_width' )

known_short_redshift		=	L_vs_z__known_short['measured z'].data
known_short_Luminosity		=	L_vs_z__known_short['Luminosity [erg/s]'].data
known_short_Luminosity_error=	L_vs_z__known_short['Luminosity_error [erg/s]'].data

Fermi_short_redshift		=	L_vs_z__Fermi_short[ 'pseudo z' ].data
Fermi_short_redshift_error	=	L_vs_z__Fermi_short[ 'pseudo z_error' ].data
Fermi_short_Luminosity		=	L_vs_z__Fermi_short['Luminosity [erg/s]'].data
Fermi_short_Luminosity_error=	L_vs_z__Fermi_short['Luminosity_error [erg/s]'].data

Swift_short_redshift		=	L_vs_z__Swift_short[ 'pseudo z' ].data
Swift_short_redshift_error	=	L_vs_z__Swift_short[ 'pseudo z_error' ].data
Swift_short_Luminosity		=	L_vs_z__Swift_short['Luminosity [erg/s]'].data
Swift_short_Luminosity_error=	L_vs_z__Swift_short['Luminosity_error [erg/s]'].data

other_short_redshift		=	L_vs_z__other_short['measured z'].data
other_short_Luminosity		=	L_vs_z__other_short['Luminosity [erg/s]'].data
other_short_Luminosity_error=	L_vs_z__other_short['Luminosity_error [erg/s]'].data

BATSE_short_redshift		=	L_vs_z__BATSE_short[ 'pseudo z' ].data
BATSE_short_redshift_error	=	L_vs_z__BATSE_short[ 'pseudo z_error' ].data
BATSE_short_Luminosity		=	L_vs_z__BATSE_short['Luminosity [erg/s]'].data
BATSE_short_Luminosity_error=	L_vs_z__BATSE_short['Luminosity_error [erg/s]'].data

FermE_short_redshift		=	L_vs_z__FermE_short[ 'pseudo z' ].data
FermE_short_redshift_error	=	L_vs_z__FermE_short[ 'pseudo z_error' ].data
FermE_short_Luminosity		=	L_vs_z__FermE_short['Luminosity [erg/s]'].data
FermE_short_Luminosity_error=	L_vs_z__FermE_short['Luminosity_error [erg/s]'].data



####################################################################################################################################################




####################################################################################################################################################



ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( known_short_redshift, known_short_Luminosity, fmt = 'd', color = 'r', label = r'$ \rm{ Fermi, known  } $' )
ax.errorbar( Fermi_short_redshift, Fermi_short_Luminosity, fmt = '.', color = 'k', label = r'$ \rm{ Fermi, pseudo : Type \, I } $' )
ax.errorbar( FermE_short_redshift, FermE_short_Luminosity, fmt = 'P', color = 'b', label = r'$ \rm{ Fermi, pseudo : Type \, II} $' )
ax.plot( z_sim, L_cut__Fermi, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Fermi_short_all.png' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Fermi_short_all.pdf' )
plt.clf()
plt.close()

print 'Fermi....\n'
#~ print np.median(   Fermi_short_redshift_error /  Fermi_short_redshift  )*100
#~ print np.median( known_short_Luminosity_error / known_short_Luminosity )*100, np.median( Fermi_short_Luminosity_error / Fermi_short_Luminosity )*100
print np.mean(   Fermi_short_redshift_error /  Fermi_short_redshift  )*100
print np.mean( known_short_Luminosity_error / known_short_Luminosity )*100, np.mean( Fermi_short_Luminosity_error / Fermi_short_Luminosity )*100
print '\n'

Fermi_short_redshift__all	=	np.append( known_short_redshift  , Fermi_short_redshift   )
Fermi_short_Luminosity__all	=	np.append( known_short_Luminosity, Fermi_short_Luminosity )
fit_the_trend( Fermi_short_redshift, Fermi_short_Luminosity/L_norm )



ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( other_short_redshift, other_short_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ Swift, known  } $' )
ax.errorbar( Swift_short_redshift, Swift_short_Luminosity, fmt = '.', color = 'k', label = r'$ \rm{ Swift, pseudo } $' )
ax.plot( z_sim, L_cut__Swift, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Swift_short_all.png' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Swift_short_all.pdf' )
plt.clf()
plt.close()

print 'Swift...\n'
#~ print np.median(   Swift_short_redshift_error /   Swift_short_redshift )*100
#~ print np.median( Swift_short_Luminosity_error / Swift_short_Luminosity )*100, np.median( other_short_Luminosity_error / other_short_Luminosity )*100
print np.mean(   Swift_short_redshift_error /   Swift_short_redshift )*100
print np.mean( Swift_short_Luminosity_error / Swift_short_Luminosity )*100, np.mean( other_short_Luminosity_error / other_short_Luminosity )*100
print '\n'

Swift_short_redshift__all	=	np.append( other_short_redshift  , Swift_short_redshift   )
Swift_short_Luminosity__all	=	np.append( other_short_Luminosity, Swift_short_Luminosity )
fit_the_trend( Swift_short_redshift, Swift_short_Luminosity/L_norm )




print 'BATSE...\n'
inds_to_delete	=	[]
for j, z in enumerate( BATSE_short_redshift ):
	array	=	np.abs( z_sim - z )
	ind		=	np.where( array == array.min() )[0]
	if ( BATSE_short_Luminosity[j] - L_cut__BATSE[ind] ) < 0 :
		inds_to_delete.append( j )
inds_to_delete	=	np.array( inds_to_delete )
print 'Number of BATSE GRBs	:	', BATSE_short_Luminosity.size
print 'BATSE GRBs, deleted	:	', inds_to_delete.size
BATSE_short_redshift			=	np.delete( BATSE_short_redshift        , inds_to_delete )
BATSE_short_redshift_error		=	np.delete( BATSE_short_redshift_error  , inds_to_delete )
BATSE_short_Luminosity			=	np.delete( BATSE_short_Luminosity      , inds_to_delete )
BATSE_short_Luminosity_error	=	np.delete( BATSE_short_Luminosity_error, inds_to_delete )


ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( BATSE_short_redshift, BATSE_short_Luminosity, fmt = '.', color = 'k', label = r'$ \rm{ BATSE, pseudo } $' )
ax.plot( z_sim, L_cut__BATSE, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--BATSE_short_all.png' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--BATSE_short_all.pdf' )
plt.clf()
plt.close()

print '\n'
#~ print np.median(   BATSE_short_redshift_error /   BATSE_short_redshift )*100
#~ print np.median( BATSE_short_Luminosity_error / BATSE_short_Luminosity )*100, np.median( other_short_Luminosity_error / other_short_Luminosity )*100
print np.mean(   BATSE_short_redshift_error /   BATSE_short_redshift )*100
print np.mean( BATSE_short_Luminosity_error / BATSE_short_Luminosity )*100, np.mean( other_short_Luminosity_error / other_short_Luminosity )*100
print '\n'

fit_the_trend( BATSE_short_redshift, BATSE_short_Luminosity/L_norm )


####################################################################################################################################################




####################################################################################################################################################



def choose( bigger, smaller ):
	
	
	indices = []
	
	for i, s in enumerate( smaller ):
		
		ind	=	np.where(bigger == s)[0]
		if ind.size != 0: indices.append( ind[0] )
	
	return np.array(indices)

other_table	=	ascii.read( './../tables/database_short--other.txt', format = 'fixed_width' )
other_name	=	other_table['name'].data
other_z		=	other_table['measured z'].data

combined_table	=	ascii.read( './../data/combined_catalogue--literature.txt', format = 'fixed_width' )
combined_name	=	combined_table['GRB'].data
combined_z		=	combined_table['z'].data

inds_to_delete	=	choose( combined_name, other_name )
inds_to_select	=	np.delete( np.arange(combined_name.size), inds_to_delete )
combined_name	=	combined_name[inds_to_select]
combined_z		=	combined_z[inds_to_select]

known_redshift	=	np.concatenate( [combined_z          , other_short_redshift] )
Fermi_redshift	=	np.concatenate( [Fermi_short_redshift, FermE_short_redshift] )
pseudo_redshift	=	np.concatenate( [Fermi_redshift, Swift_short_redshift, BATSE_short_redshift] )


print 'total , Swift	:	',Swift_short_redshift__all.size, ' = ', other_short_redshift.size, '(other) + ', Swift_short_redshift.size, '(pseudo).'
print 'known , Swift +	:	', known_redshift.size, '\n'
print 'total , Fermi 	:	', Fermi_redshift.size, ' = ', Fermi_short_redshift.size, '(with spectra) + ', FermE_short_redshift.size, '(without).'
print '        BATSE 	:	', BATSE_short_redshift.size
print 'pseudo, Swift 	:	', Swift_short_redshift.size
print 'total , pseudo	:	', pseudo_redshift.size

from scipy import stats
print '\n\n\n\n\n\n\n'


z_min = 0.0 ; z_max = 10.0 ; z_bin = 5e-1

hist	=	mf.my_histogram_according_to_given_boundaries( known_redshift      , 1.0*z_bin, z_min, z_max )	;	kx	=	hist[0]	;	ky	=	hist[1]	;	norm	=	ky.sum()
hist	=	mf.my_histogram_according_to_given_boundaries( BATSE_short_redshift, 1.0*z_bin, z_min, z_max )	;	bx	=	hist[0]	;	by	=	hist[1]	;	by	=	1.0 * norm * ( by/by.sum() )
hist	=	mf.my_histogram_according_to_given_boundaries( Fermi_redshift      , 1.0*z_bin, z_min, z_max )	;	fx	=	hist[0]	;	fy	=	hist[1]	;	fy	=	1.0 * norm * ( fy/fy.sum() )
hist	=	mf.my_histogram_according_to_given_boundaries( Swift_short_redshift, 2.0*z_bin, z_min, z_max )	;	sx	=	hist[0]	;	sy	=	hist[1]	;	sy	=	0.5 * norm * ( sy/sy.sum() )

ky	=	ky / ky.sum()	;	ky	=	np.cumsum(ky)
by	=	by / by.sum()	;	by	=	np.cumsum(by)
fy	=	fy / fy.sum()	;	fy	=	np.cumsum(fy)
sy	=	sy / sy.sum()	;	sy	=	np.cumsum(sy)

print stats.ks_2samp(ky, by)
print stats.ks_2samp(ky, fy)
print stats.ks_2samp(ky, sy)
print '\n\n'


plt.xlabel( r'$ z $', fontsize = size_font )
plt.ylabel( r'$ \rm{normalized} \;\; N(<z) $', fontsize = size_font )
plt.step( kx, ky, color = 'k', linestyle = '-' , label = r'$ \rm{known} $' )
plt.step( bx, by, color = 'y', linestyle = '--', label = r'$ BATSE $' )
plt.step( fx, fy, color = 'r', linestyle = '-.', label = r'$ Fermi $' )
plt.step( sx, sy, color = 'b', linestyle = ':' , label = r'$ Swift $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/redshift_distributions--cumulative.png' )
plt.savefig( './../plots/pseudo_calculations/redshift_distributions--cumulative.pdf' )
plt.clf()
plt.close()


z_min = 0.0 ; z_max = 1.2 ; z_bin = 1e-1
print 'Number of known below z_max:	', np.where( known_redshift < z_max )[0].size
print 'Number of known below z_max:	', np.where( Fermi_redshift < z_max )[0].size
print 'Number of known below z_max:	', np.where( BATSE_short_redshift < z_max )[0].size
print 'Number of known below z_max:	', np.where( Swift_short_redshift < z_max )[0].size
print '\n\n'

hist	=	mf.my_histogram_according_to_given_boundaries( known_redshift      , 1.0*z_bin, z_min, z_max )	;	kx	=	hist[0]	;	ky	=	hist[1]
hist	=	mf.my_histogram_according_to_given_boundaries( BATSE_short_redshift, 1.0*z_bin, z_min, z_max )	;	bx	=	hist[0]	;	by	=	hist[1]
hist	=	mf.my_histogram_according_to_given_boundaries( Fermi_redshift      , 1.0*z_bin, z_min, z_max )	;	fx	=	hist[0]	;	fy	=	hist[1]
hist	=	mf.my_histogram_according_to_given_boundaries( Swift_short_redshift, 2.0*z_bin, z_min, z_max )	;	sx	=	hist[0]	;	sy	=	hist[1]

ky	=	ky / ky.sum()	;	ky	=	np.cumsum(ky)
by	=	by / by.sum()	;	by	=	np.cumsum(by)
fy	=	fy / fy.sum()	;	fy	=	np.cumsum(fy)
sy	=	sy / sy.sum()	;	sy	=	np.cumsum(sy)

print stats.ks_2samp(ky, by)
print stats.ks_2samp(ky, fy)
print stats.ks_2samp(ky, sy)

plt.xlabel( r'$ z $', fontsize = size_font )
plt.ylabel( r'$ \rm{normalized} \;\; N(<z) $', fontsize = size_font )
plt.step( kx, ky, color = 'k', linestyle = '-' , label = r'$ \rm{known} $' )
plt.step( bx, by, color = 'y', linestyle = '--', label = r'$ BATSE $' )
plt.step( fx, fy, color = 'r', linestyle = '-.', label = r'$ Fermi $' )
plt.step( sx, sy, color = 'b', linestyle = ':' , label = r'$ Swift $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/redshift_distributions--cumulative--truncated.png' )
plt.savefig( './../plots/pseudo_calculations/redshift_distributions--cumulative--truncated.pdf' )
plt.clf()
plt.close()
