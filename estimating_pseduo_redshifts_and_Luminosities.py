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
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9


A___Tsutsui		=	2.927		#	best-fit from Tsutsui-2013
eta_Tsutsui		=	1.590		#	best-fit from Tsutsui-2013
A___mybestfit	=	3.031		#	my best-fit
eta_mybestfit	=	1.725		#	my best-fit

A	=	A___mybestfit
eta	=	eta_mybestfit


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	1e+1 #	for the purposes of plotting
x_in_keV_min	=	1e01	;	x_in_keV_max	=	5e04	#	Ep(1+z), min & max.
y_in_eps_min	=	1e48	;	y_in_eps_max	=	1e55	#	L_iso  , min & max.


####################################################################################################################################################






####################################################################################################################################################
########	Defining the functions.


def choose( bigger, smaller ):
	
	
	indices = []
	
	for i, s in enumerate( smaller ):
		ind	=	np.where(bigger == s)[0][0]		# the index is of the bigger array.
		indices.append( ind )
	
	
	return np.array(indices)



k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
k_BATSE		=	k_table['k_BATSE'].data
term_Fermi	=	k_table['term_Fermi'].data
term_Swift	=	k_table['term_Swift'].data
term_BATSE	=	k_table['term_BATSE'].data
z_bin		=	np.mean( np.diff(z_sim) )


numerator_F		=	4*P  *  ( dL_sim**2 )
denominator_F	=	(1+z_sim)**eta

denominator__delta_pseudo_z__Fermi		=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )  +  (  (eta+2) / (1+z_sim)  ) + term_Fermi
numerator_F__Fermi	=	k_Fermi  *  numerator_F
F_Fermi		=	numerator_F__Fermi / denominator_F
F_Fermi		=	F_Fermi * (cm_per_Mpc**2) / L_norm

denominator__delta_pseudo_z__Swift		=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )  +  (  (eta+2) / (1+z_sim)  ) + term_Swift
numerator_F__Swift	=	k_Swift  *  numerator_F
F_Swift		=	numerator_F__Swift / denominator_F
F_Swift		=	F_Swift * (cm_per_Mpc**2) / L_norm
F_Swift		=	F_Swift * erg_per_keV

denominator__delta_pseudo_z__BATSE		=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )  +  (  (eta+2) / (1+z_sim)  ) + term_BATSE
numerator_F__BATSE	=	k_BATSE  *  numerator_F
F_BATSE		=	numerator_F__BATSE / denominator_F
F_BATSE		=	F_BATSE * (cm_per_Mpc**2) / L_norm
F_BATSE		=	F_BATSE * erg_per_keV




def estimate_pseudo_redshift_and_Luminosity__Fermi( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, alpha, beta ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
		
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Fermi - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__Fermi[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', np.min(pseudo_redshifts), np.max(pseudo_redshifts), '\n'
	
	
	Epeak_into_oneplusz	=	Epeak_in_MeV*(1+pseudo_redshifts)
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		al		=	alpha[j]
		be		=	beta[j]
		Ep_keV	=	Epeak_in_MeV[j] * 1e3
		Luminosity[j]	=	sf.Liso_with_known_spectral_parameters__Fermi( ep, al, be, Ep_keV, z )
	Luminosity			=	Luminosity * ( cm_per_Mpc**2 )
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshift)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', np.min(percentage_error), np.max(percentage_error)
	print 'Percentage error, mean		:	', percentage_error.mean()
	print 'Luminosity error, mean:			', 100 * np.median(Luminosity_error/Luminosity)
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error


def estimate_pseudo_redshift_and_Luminosity__Swift( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )		#	defined for each GRB
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
	
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Swift - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__Swift[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', np.min(pseudo_redshifts), np.max(pseudo_redshifts), '\n'
	
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( ep, z )
	Luminosity			=	Luminosity * (cm_per_Mpc**2) * erg_per_keV
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshifts)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', percentage_error.min(), percentage_error.max()
	print 'Percentage error, mean		:	', np.median(percentage_error)
	print 'Luminosity error, mean:			', 100 * np.median(Luminosity_error/Luminosity)
	inds_huge_errors	=	np.where( percentage_error >= percentage_cutoff )[0]
	print '# of GRBs with errors > {0:d}%	:	{1:d}'.format(percentage_cutoff, inds_huge_errors.size)
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error


def estimate_pseudo_redshift_and_Luminosity__BATSE( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )		#	defined for each GRB
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
	
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_BATSE - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__BATSE[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', np.min(pseudo_redshifts), np.max(pseudo_redshifts.max), '\n'
	
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__BATSE( ep, z )
	Luminosity			=	Luminosity * (cm_per_Mpc**2) * erg_per_keV
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshifts)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	inds_to_delete	=	np.where( pseudo_redshifts >= 20 )[0]
	print '# of GRBs with redshift > 20	:	', inds_to_delete.size
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', np.min(percentage_error), np.max(percentage_error)
	print 'Percentage error, mean		:	', np.median(percentage_error)
	print 'Luminosity error, mean:			', 100 * np.median(Luminosity_error/Luminosity)
	inds_huge_errors	=	np.where( percentage_error >= percentage_cutoff )[0]
	print '# of GRBs with errors > {0:d}%	:	{1:d}'.format(percentage_cutoff, inds_huge_errors.size)
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error


def estimate_pseudo_redshift_and_Luminosity__Fermi_exclusive( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )		#	defined for each GRB
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
	
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Fermi - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__Fermi[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', pseudo_redshifts.min(), pseudo_redshifts.max(), '\n'
	
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__Fermi( ep, z )
	Luminosity			=	Luminosity * (cm_per_Mpc**2)
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshifts)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', percentage_error.min(), percentage_error.max()
	print 'Percentage error, mean		:	', percentage_error.mean()
	print 'Luminosity error, mean:			', 100 * (Luminosity_error/Luminosity).mean()
	inds_huge_errors	=	np.where( percentage_error >= percentage_cutoff )[0]
	print '# of GRBs with errors > {0:d}%	:	{1:d}'.format(percentage_cutoff, inds_huge_errors.size)
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error



####################################################################################################################################################






####################################################################################################################################################
########	Reading the data.



Fermi_GRBs_table			=	ascii.read( './../data/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_name					=	Fermi_GRBs_table['Fermi name'].data
Fermi_T90					=	Fermi_GRBs_table['GBM T90'].data
Fermi_T90_error				=	Fermi_GRBs_table['GBM T90_error'].data
Fermi_flux					=	Fermi_GRBs_table['GBM flux'].data
Fermi_flux_error			=	Fermi_GRBs_table['GBM flux_error'].data
Fermi_Epeak      			=	Fermi_GRBs_table['Epeak'].data
Fermi_Epeak_error			=	Fermi_GRBs_table['Epeak_error'].data
Fermi_alpha					=	Fermi_GRBs_table['alpha'].data
Fermi_alpha_error			=	Fermi_GRBs_table['alpha_error'].data
Fermi_beta					=	Fermi_GRBs_table['beta'].data
Fermi_beta_error			=	Fermi_GRBs_table['beta_error'].data
Fermi_num					=	Fermi_name.size



Swift_all_GRBs_table		=	ascii.read( './../data/Swift_GRBs--all.txt', format = 'fixed_width' )
Swift_all_name				=	Swift_all_GRBs_table['Swift name'].data
Swift_all_T90				=	Swift_all_GRBs_table['BAT T90'].data
Swift_all_flux				=	Swift_all_GRBs_table['BAT Phoflux'].data
Swift_all_flux_error		=	Swift_all_GRBs_table['BAT Phoflux_error'].data
Swift_all_num				=	Swift_all_name.size

Swift_wkr_GRBs_table		=	ascii.read( './../data/Swift_GRBs--wkr.txt', format = 'fixed_width' )
Swift_wkr_name				=	Swift_wkr_GRBs_table['Swift name'].data
Swift_wkr_redhsift			=	Swift_wkr_GRBs_table['redshift'].data
Swift_wkr_T90				=	Swift_wkr_GRBs_table['BAT T90'].data
Swift_wkr_flux				=	Swift_wkr_GRBs_table['BAT Phoflux'].data
Swift_wkr_flux_error		=	Swift_wkr_GRBs_table['BAT Phoflux_error'].data
Swift_wkr_num				=	Swift_wkr_name.size
Swift_wkr_num				=	Swift_wkr_name.size



common_all_GRBs_table		=	ascii.read( './../data/common_GRBs--all.txt', format = 'fixed_width' )
common_all_ID				=	common_all_GRBs_table['common ID'].data
common_all_Swift_name		=	common_all_GRBs_table['Swift name'].data
common_all_Fermi_name		=	common_all_GRBs_table['Fermi name'].data
common_all_Swift_T90		=	common_all_GRBs_table['BAT T90'].data
common_all_Fermi_T90		=	common_all_GRBs_table['GBM T90'].data
common_all_Fermi_T90_error	=	common_all_GRBs_table['GBM T90_error'].data
common_all_Fermi_flux		=	common_all_GRBs_table['GBM flux'].data
common_all_Fermi_flux_error	=	common_all_GRBs_table['GBM flux_error'].data
common_all_Epeak			=	common_all_GRBs_table['Epeak'].data				#	in keV.
common_all_Epeak_error		=	common_all_GRBs_table['Epeak_error'].data		#	in keV.
common_all_alpha			=	common_all_GRBs_table['alpha'].data
common_all_alpha_error		=	common_all_GRBs_table['alpha_error'].data
common_all_beta				=	common_all_GRBs_table['beta'].data
common_all_beta_error		=	common_all_GRBs_table['beta_error'].data
common_all_num				=	common_all_ID.size

common_wkr_GRBs_table		=	ascii.read( './../data/common_GRBs--wkr.txt', format = 'fixed_width' )
common_wkr_ID				=	common_wkr_GRBs_table['common ID'].data
common_wkr_Swift_name		=	common_wkr_GRBs_table['Swift name'].data
common_wkr_Fermi_name		=	common_wkr_GRBs_table['Fermi name'].data
common_wkr_Swift_T90		=	common_wkr_GRBs_table['BAT T90'].data
common_wkr_Fermi_T90		=	common_wkr_GRBs_table['GBM T90'].data
common_wkr_Fermi_T90_error	=	common_wkr_GRBs_table['GBM T90_error'].data
common_wkr_redshift			=	common_wkr_GRBs_table['redshift'].data
common_wkr_Fermi_flux		=	common_wkr_GRBs_table['GBM flux'].data
common_wkr_Fermi_flux_error	=	common_wkr_GRBs_table['GBM flux_error'].data
common_wkr_Epeak			=	common_wkr_GRBs_table['Epeak'].data				#	in keV.
common_wkr_Epeak_error		=	common_wkr_GRBs_table['Epeak_error'].data		#	in keV.
common_wkr_alpha			=	common_wkr_GRBs_table['alpha'].data
common_wkr_alpha_error		=	common_wkr_GRBs_table['alpha_error'].data
common_wkr_beta				=	common_wkr_GRBs_table['beta'].data
common_wkr_beta_error		=	common_wkr_GRBs_table['beta_error'].data
common_wkr_Luminosity		=	common_wkr_GRBs_table['Luminosity'].data
common_wkr_Luminosity_error	=	common_wkr_GRBs_table['Luminosity_error'].data
common_wkr_num				=	common_wkr_ID.size



BATSE_GRBs_table			=	ascii.read( './../tables/BATSE_GRBs--measured.txt', format = 'fixed_width' )
BATSE_name					=	BATSE_GRBs_table['name'].data
BATSE_T90					=	BATSE_GRBs_table['T90'].data
BATSE_T90_error				=	BATSE_GRBs_table['T90_error'].data
BATSE_flux					=	BATSE_GRBs_table['flux'].data
BATSE_flux_error			=	BATSE_GRBs_table['flux_error'].data
BATSE_num					=	BATSE_name.size
print 'Number of BATSE GRBs	:	' , BATSE_num
#~ inds						=	np.where( BATSE_flux > BATSE_sensitivity )[0]
inds						=	np.where( BATSE_flux != 0 )[0]
BATSE_name					=	BATSE_name[inds]
BATSE_T90					=	BATSE_T90[inds]
BATSE_T90_error				=	BATSE_T90_error[inds]
BATSE_flux					=	BATSE_flux[inds]
BATSE_flux_error			=	BATSE_flux_error[inds]
BATSE_num					=	BATSE_name.size
print 'Number of BATSE GRBs	:	' , BATSE_num



Fermi_exclusive_GRBs_table			=	ascii.read( './../tables/Fermi_GRBs--exclusive.txt', format = 'fixed_width' )
Fermi_exclusive_name				=	Fermi_exclusive_GRBs_table['name'].data	
Fermi_exclusive_T90					=	Fermi_exclusive_GRBs_table['T90'].data
Fermi_exclusive_T90_error			=	Fermi_exclusive_GRBs_table['T90_error'].data
Fermi_exclusive_flux				=	Fermi_exclusive_GRBs_table['flux [erg.cm^{-2}.s^{-1}]'].data
Fermi_exclusive_flux_error			=	Fermi_exclusive_GRBs_table['flux_error [erg.cm^{-2}.s^{-1}]'].data
Fermi_exclusive_Epeak_in_MeV		=	Fermi_exclusive_GRBs_table['Epeak (siml) [MeV]'].data
Fermi_exclusive_Epeak_in_MeV_error	=	Fermi_exclusive_GRBs_table['Epeak_error (siml) [MeV]'].data
Fermi_exclusive_num					=	Fermi_exclusive_name.size

print '\n\n'
print Fermi_exclusive_flux.min(), Fermi_exclusive_flux.max()
print Fermi_flux.min(), Fermi_flux.max()
print '\n\n'



####################################################################################################################################################







####################################################################################################################################################
########	For the Fermi GRBs, including those common with Swift (since they have spectra), except those with known redshifts (L already known).





####	First for the long ones.

##	Finding all the long GRBs.
print 'Number of common GRBs				:	', common_all_num

inds_long_in_universal_common_sample_by_applying_Swift_criterion		=	np.where( common_all_Swift_T90 >= T90_cut )[0]	# these indices run over the sample of all common GRBs (i.e. with/without redshift).
print 'Number of long ones amongst them 		:	', inds_long_in_universal_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion			=	common_all_Fermi_name[inds_long_in_universal_common_sample_by_applying_Swift_criterion]
inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion		=	choose( Fermi_name, Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.

print 'Total number of GRBs in the Fermi sample	:	', Fermi_num
inds_in_Fermi_common_all		=	choose( Fermi_name, common_all_Fermi_name )	# these indices run over the universal Fermi sample.
print 'Number of common GRBs				:	', inds_in_Fermi_common_all.size
inds_in_Fermi_uncommon_all		=	np.delete( np.arange(Fermi_num), inds_in_Fermi_common_all )	# these indices run over the universal Fermi sample.
print 'Out of which those detected only by Fermi	:	', inds_in_Fermi_uncommon_all.size

Fermi_T90_uncommon_all	=	Fermi_T90[ inds_in_Fermi_uncommon_all]
Fermi_name_uncommon_all	=	Fermi_name[inds_in_Fermi_uncommon_all]

inds_long_in_Fermi_only_sample				=	np.where( Fermi_T90_uncommon_all >= T90_cut )[0]	# these indices run over the Fermi-only sample (with/without redshift).
Fermi_name_for_long_in_Fermi_only_sample	=	Fermi_name_uncommon_all[inds_long_in_Fermi_only_sample]
print 'Long in Fermi-only sample, by Fermi criterion	:	', inds_long_in_Fermi_only_sample.size

inds_long_in_Fermi_full_sample	=	choose( Fermi_name, Fermi_name_for_long_in_Fermi_only_sample )	# these indices run over the universal Fermi sample.

inds_long_in_Fermi	=	np.union1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion )		# these indices run over the universal Fermi sample.
print 'Total number of long GRBs in the Fermi sample	:	', inds_long_in_Fermi.size



##	Finding only the long GRBs without redshift.
print '\n\n'
print 'Number of common GRBs with known redshift	:	', common_wkr_num

inds_long_in_redshift_common_sample_by_applying_Swift_criterion		=	np.where( common_wkr_Swift_T90 >= T90_cut )[0]	# these indices run over the sample of common GRBs with known redshift.
print 'Number of long ones amongst them 		:	', inds_long_in_redshift_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_long_in_redshift_common_sample_by_applying_Swift_criterion			=	common_wkr_Fermi_name[inds_long_in_redshift_common_sample_by_applying_Swift_criterion]
inds_amongst_common_all_for_those_wkr	=	choose( Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion, Fermi_name_for_long_in_redshift_common_sample_by_applying_Swift_criterion ) 

inds_without_redshift_amongst_common_sample		=	np.delete( np.arange(inds_long_in_universal_common_sample_by_applying_Swift_criterion.size), inds_amongst_common_all_for_those_wkr )	# these indices run over all the common GRBs that are long (only, by Swift criterion).
Fermi_name_for_common_long_GRBs_without_redshift	=	Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion[inds_without_redshift_amongst_common_sample]
print 'Common GRBs with unknown redshift		:	', Fermi_name_for_common_long_GRBs_without_redshift.size

inds_in_Fermi_for_common_long_GRBs_without_redshift	=	choose( Fermi_name, Fermi_name_for_common_long_GRBs_without_redshift )	# these indices run over the universal Fermi sample.

inds_long_in_Fermi_without_redshift	=	np.union1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_common_long_GRBs_without_redshift )		# these indices run over the universal Fermi sample.
print 'Total number of Fermi l-GRBs without redshift	:	', inds_long_in_Fermi_without_redshift.size
print '\n\n'

print '\n\n\n\n'









####	Similarly for short GRBs in Fermi.
print 'Number of common GRBs				:	', common_all_num

inds_short_in_universal_common_sample_by_applying_Swift_criterion		=	np.where( common_all_Swift_T90 < T90_cut )[0]	# these indices run over the sample of all common GRBs (i.e. with/without redshift).
print 'Number of short ones amongst them 		:	', inds_short_in_universal_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion			=	common_all_Fermi_name[inds_short_in_universal_common_sample_by_applying_Swift_criterion]
inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion	=	choose( Fermi_name, Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.

print 'Total number of GRBs in the Fermi sample	:	', Fermi_num
print 'Out of which those detected only by Fermi	:	', inds_in_Fermi_uncommon_all.size


inds_short_in_Fermi_only_sample				=	np.where( Fermi_T90_uncommon_all < T90_cut )[0]	# these indices run over the Fermi-only sample (with/without redshift).
Fermi_name_for_short_in_Fermi_only_sample	=	Fermi_name_uncommon_all[inds_short_in_Fermi_only_sample]
print 'Short in Fermi-only sample, by Fermi criterion	:	', inds_short_in_Fermi_only_sample.size

inds_short_in_Fermi_full_sample	=	choose( Fermi_name, Fermi_name_for_short_in_Fermi_only_sample )	# these indices run over the universal Fermi sample.

inds_short_in_Fermi				=	np.union1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.
print 'Total number of short GRBs in the Fermi sample	:	', inds_short_in_Fermi.size



##	Finding only the short GRBs without redshift.
print '\n\n'
print 'Number of common GRBs with known redshift	:	', common_wkr_num

inds_short_in_redshift_common_sample_by_applying_Swift_criterion		=	np.where( common_wkr_Swift_T90 < T90_cut )[0]	# these indices run over the sample of common GRBs with known redshift.
print 'Number of short ones amongst them 		:	', inds_short_in_redshift_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_short_in_redshift_common_sample_by_applying_Swift_criterion		=	common_wkr_Fermi_name[inds_short_in_redshift_common_sample_by_applying_Swift_criterion]
inds_amongst_common_all_for_those_wkr	=	choose( Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion, Fermi_name_for_short_in_redshift_common_sample_by_applying_Swift_criterion ) 

inds_without_redshift_amongst_common_sample		=	np.delete( np.arange(inds_short_in_universal_common_sample_by_applying_Swift_criterion.size), inds_amongst_common_all_for_those_wkr )	# these indices run over all the common GRBs that are long (only, by Swift criterion).
Fermi_name_for_common_short_GRBs_without_redshift	=	Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion[inds_without_redshift_amongst_common_sample]
print 'Common GRBs with unknown redshift		:	', Fermi_name_for_common_short_GRBs_without_redshift.size

inds_in_Fermi_for_common_short_GRBs_without_redshift	=	choose( Fermi_name, Fermi_name_for_common_short_GRBs_without_redshift )	# these indices run over the universal Fermi sample..

inds_short_in_Fermi_without_redshift	=	np.union1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_common_short_GRBs_without_redshift )		# these indices run over the universal Fermi sample.
print 'Total number of Fermi s-GRBs without redshift	:	', inds_short_in_Fermi_without_redshift.size
print '\n\n'

print '\n\n\n\n'





Fermi_short_name				=	Fermi_name[inds_short_in_Fermi_without_redshift]
Fermi_short_flux				=	Fermi_flux[inds_short_in_Fermi_without_redshift]
Fermi_short_flux_error			=	Fermi_flux_error[inds_short_in_Fermi_without_redshift]
Fermi_short_Epeak_in_keV		=	Fermi_Epeak[inds_short_in_Fermi_without_redshift]		# in keV.
Fermi_short_Epeak_in_keV_error	=	Fermi_Epeak_error[inds_short_in_Fermi_without_redshift]	# same as above.
Fermi_short_Epeak_in_MeV		=	1e-3 * Fermi_short_Epeak_in_keV			# in MeV.
Fermi_short_Epeak_in_MeV_error	=	1e-3 * Fermi_short_Epeak_in_keV_error	# same as above.
Fermi_short_alpha				=	Fermi_alpha[inds_short_in_Fermi_without_redshift]
Fermi_short_beta				=	Fermi_beta[inds_short_in_Fermi_without_redshift]

print '\n\n'
print '#### Fermi short GRBs ####', '\n'
print 'Number of GRBs put in		:	', Fermi_short_flux.size, '\n'
Fermi_short_name, Fermi_short_flux, Fermi_short_flux_error, Fermi_short_Epeak_in_MeV, Fermi_short_Epeak_in_MeV_error, Fermi_short_pseudo_redshift, Fermi_short_pseudo_redshift_error, Fermi_short_Luminosity, Fermi_short_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Fermi( Fermi_short_name, Fermi_short_flux, Fermi_short_flux_error, Fermi_short_Epeak_in_MeV, Fermi_short_Epeak_in_MeV_error, Fermi_short_alpha, Fermi_short_beta )
print 'Number of GRBs selected		:	', Fermi_short_flux.size
print '\n\n\n\n'


####################################################################################################################################################








####################################################################################################################################################
########	For the Swift GRBs, excluding those common with Fermi (since they have spectra), and those with known redshifts.




inds_common		=	choose( Swift_all_name, common_all_Swift_name )
inds_wkr		=	choose( Swift_all_name, Swift_wkr_name )
inds_to_delete	=	np.union1d( inds_common, inds_wkr )
inds_exclusively_Swift_GRBs_without_redshift	=	np.delete( np.arange(Swift_all_num), inds_to_delete )
Swift_num	=	inds_exclusively_Swift_GRBs_without_redshift.size

print '\n\n\n\n\n\n\n\n'
print ' #### Swift GRBs ####', '\n'
print '# of common GRBs		:	', inds_common.size
print '# of GRBs with redshift		:	', inds_wkr.size
print '# of common amongst these	:	', np.intersect1d( inds_common, inds_wkr ).size
print '# to be finally deleted		:	', inds_to_delete.size, '\n'
print '# of Swift GRBs, total		:	', Swift_all_num
print '# to be selected		:	', Swift_num

Swift_name			=	Swift_all_name[inds_exclusively_Swift_GRBs_without_redshift]
Swift_T90			=	Swift_all_T90[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux			=	Swift_all_flux[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux_error	=	Swift_all_flux_error[inds_exclusively_Swift_GRBs_without_redshift]



hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_Epeak), 0.125, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]

Swift_Epeak_in_keV			=	np.random.normal( f0, f1, Swift_num )
Swift_Epeak_in_keV			=	10**Swift_Epeak_in_keV			#	in keV
Swift_Epeak_in_MeV			=	1e-3 * Swift_Epeak_in_keV		#	in MeV
Swift_Epeak_in_MeV_error	=	np.zeros( Swift_Epeak_in_MeV.size )

#~ plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
#~ plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
#~ plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/pseudo_calculations/Fermi--Ep_distribution.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Swift_Epeak_in_keV), 0.125, 1, 4 )	;	sx	=	hist[0]	;	sy	=	hist[1]
#~ plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
#~ plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
#~ plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
#~ plt.step( sx, sy * (hy.max()/sy.max()), color = 'b', label = r'$ Swift \rm{ , \; simulated } $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/pseudo_calculations/Swift--Ep_distribution--simulated_1.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ plt.title( r'$ Swift $', fontsize = size_font )
#~ plt.hist( Swift_Epeak_in_keV, bins = np.logspace(1, 4, 20) )
#~ plt.gca().set_xscale('log')
#~ plt.xlabel( r'$ E_p \; \rm{ [keV] } $', fontsize = size_font )
#~ plt.savefig( './../plots/pseudo_calculations/Swift--Ep_distribution--simulated_2.png' )
#~ plt.clf()
#~ plt.close()





inds_long_in_Swift	=	np.where( Swift_T90 >= T90_cut )[0]
inds_short_in_Swift	=	np.delete( np.arange(Swift_num), inds_long_in_Swift )

Swift_long_name					=	Swift_name[inds_long_in_Swift]
Swift_long_T90					=	Swift_T90[inds_long_in_Swift]
Swift_long_flux					=	Swift_flux[inds_long_in_Swift]
Swift_long_flux_error			=	Swift_flux_error[inds_long_in_Swift]
Swift_long_Epeak_in_MeV			=	Swift_Epeak_in_MeV[inds_long_in_Swift]
Swift_long_Epeak_in_MeV_error	=	Swift_Epeak_in_MeV_error[inds_long_in_Swift]
Swift_long_num					=	Swift_long_name.size

Swift_short_name				=	Swift_name[inds_short_in_Swift]
Swift_short_T90					=	Swift_T90[inds_short_in_Swift]
Swift_short_flux				=	Swift_flux[inds_short_in_Swift]
Swift_short_flux_error			=	Swift_flux_error[inds_short_in_Swift]
Swift_short_Epeak_in_MeV		=	Swift_Epeak_in_MeV[inds_short_in_Swift]
Swift_short_Epeak_in_MeV_error	=	Swift_Epeak_in_MeV_error[inds_short_in_Swift]
Swift_short_num					=	Swift_short_name.size


print 'Out of which, # of long  GRBs	:	', Swift_long_num
print '                   short GRBs	:	', Swift_short_num



print '\n\n'
print '#### Swift short GRBs ####', '\n'
print 'Number of GRBs put in		:	', Swift_short_num, '\n'
Swift_short_name, Swift_short_flux, Swift_short_flux_error, Swift_short_Epeak_in_MeV, Swift_short_Epeak_in_MeV_error, Swift_short_pseudo_redshift, Swift_short_pseudo_redshift_error, Swift_short_Luminosity, Swift_short_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Swift( Swift_short_name, Swift_short_flux, Swift_short_flux_error, Swift_short_Epeak_in_MeV, Swift_short_Epeak_in_MeV_error )
print 'Number of GRBs selected		:	', Swift_short_flux.size
print '\n\n\n\n'



####################################################################################################################################################












####################################################################################################################################################
########	For the Swift-only GRBs with known redshifts, called "other" GRBs.




inds_in_Swift_wkr_for_common_wkr	=	choose( Swift_wkr_name, common_wkr_Swift_name )
inds_exclusively_Swift_GRBs_with_redshift	=	np.delete( np.arange(Swift_wkr_num), inds_in_Swift_wkr_for_common_wkr )
other_num	=	inds_exclusively_Swift_GRBs_with_redshift.size

print '\n\n\n\n\n\n\n\n'
print ' #### other GRBs ####', '\n'
print '# of Swift  GRBs wkr	:	', Swift_wkr_num
print '# of common GRBs wkr	:	', common_wkr_num
print '# of Swift-only wkr	:	', other_num

other_Swift_name			=	Swift_wkr_name[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_redshift		=	Swift_wkr_redhsift[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_T90				=	Swift_wkr_T90[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux			=	Swift_wkr_flux[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux_error		=	Swift_wkr_flux_error[inds_exclusively_Swift_GRBs_with_redshift]




other_Luminosity	=	other_Swift_redshift.copy()
for j, z in enumerate(other_Swift_redshift):
	ep		=	other_Swift_flux[j]
	other_Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( ep, z )
other_Luminosity		=	other_Luminosity * (cm_per_Mpc**2) * erg_per_keV
other_Luminosity_error	=	other_Luminosity * (other_Swift_flux_error/other_Swift_flux)



inds_other_long		=	np.where( other_Swift_T90 >= T90_cut )[0]
inds_other_short	=	np.delete( np.arange(other_num), inds_other_long )

other_long_name					=	other_Swift_name[inds_other_long]
other_long_redshift				=	other_Swift_redshift[inds_other_long]
other_long_Luminosity			=	other_Luminosity[inds_other_long]
other_long_Luminosity_error		=	other_Luminosity_error[inds_other_long]
other_long_num					=	inds_other_long.size

other_short_name				=	other_Swift_name[inds_other_short]
other_short_redshift			=	other_Swift_redshift[inds_other_short]
other_short_Luminosity			=	other_Luminosity[inds_other_short]
other_short_Luminosity_error	=	other_Luminosity_error[inds_other_short]
other_short_num					=	inds_other_short.size


print 'Out of which, # of long	:	', other_long_num
print '                  short	:	', other_short_num
print '\n\n\n\n'



####################################################################################################################################################









####################################################################################################################################################
########	For the common GRBs with known redshifts.


inds_common_long	=	np.where( common_wkr_Swift_T90 >= T90_cut )[0]
inds_common_short	=	np.delete( np.arange(common_wkr_num), inds_common_long )
print '\n\n\n\n\n\n\n\n'
print ' #### common GRBs ####'
print common_wkr_Fermi_name[inds_common_short]
print '\n\n\n\n'

known_long_Luminosity	=	common_wkr_Luminosity[inds_common_long ] * L_norm
known_short_Luminosity	=	common_wkr_Luminosity[inds_common_short] * L_norm


####################################################################################################################################################









####################################################################################################################################################
########	For the BATSE GRBs.




hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_Epeak), 0.125, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]

BATSE_Epeak_in_keV			=	np.random.normal( f0, f1, BATSE_num )
BATSE_Epeak_in_keV			=	10**BATSE_Epeak_in_keV			#	in keV
BATSE_Epeak_in_MeV			=	1e-3 * BATSE_Epeak_in_keV		#	in MeV
BATSE_Epeak_in_MeV_error	=	np.zeros( BATSE_Epeak_in_MeV.size )

#~ hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(BATSE_Epeak_in_keV), 0.125, 1, 4 )	;	bx	=	hist[0]	;	by	=	hist[1]
#~ plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
#~ plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
#~ plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
#~ plt.step( bx, by * (hy.max()/by.max()), color = 'y', label = r'$ BATSE \rm{ , \; simulated } $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/pseudo_calculations/BATSE--Ep_distribution--simulated_1.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ plt.title( r'$ BATSE $', fontsize = size_font )
#~ plt.hist( BATSE_Epeak_in_keV, bins = np.logspace(1, 4, 20) )
#~ plt.gca().set_xscale('log')
#~ plt.xlabel( r'$ E_p \; \rm{ [keV] } $', fontsize = size_font )
#~ plt.savefig( './../plots/pseudo_calculations/BATSE--Ep_distribution--simulated_2.png' )
#~ plt.clf()
#~ plt.close()




hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(BATSE_T90), 0.25, -3, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ T_{90} $' + r'$ \rm{ \, [s] \, ) } $' , fontsize = size_font )
plt.step( hx, hy, color = 'k' )
plt.savefig( './../plots/pseudo_calculations/BATSE--T90_distribution.png' )
plt.clf()
plt.close()

inds							=	np.where( BATSE_T90 < T90_cut )
BATSE_short_name				=	BATSE_name[inds]
BATSE_short_T90					=	BATSE_T90[inds]
BATSE_short_T90_error			=	BATSE_T90_error[inds]
BATSE_short_flux				=	BATSE_flux[inds]
BATSE_short_flux_error			=	BATSE_flux_error[inds]
BATSE_short_Epeak_in_MeV		=	BATSE_Epeak_in_MeV[inds]
BATSE_short_Epeak_in_MeV_error	=	BATSE_Epeak_in_MeV_error[inds]
BATSE_short_num					=	BATSE_name.size



print '\n\n\n\n'
print '#### BATSE short GRBs ####', '\n'
print 'Number of GRBs put in		:	', BATSE_short_flux.size, '\n'
BATSE_short_name, BATSE_short_flux, BATSE_short_flux_error, BATSE_short_Epeak_in_MeV, BATSE_short_Epeak_in_MeV_error, BATSE_short_pseudo_redshift, BATSE_short_pseudo_redshift_error, BATSE_short_Luminosity, BATSE_short_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__BATSE( BATSE_short_name, BATSE_short_flux, BATSE_short_flux_error, BATSE_short_Epeak_in_MeV, BATSE_short_Epeak_in_MeV_error )
print 'Number of GRBs selected		:	', BATSE_short_flux.size
print '\n\n\n\n'


####################################################################################################################################################







####################################################################################################################################################
########	For the exclsuive Fermi GRBs (i.e. not common to Swift) without spectral parameter measurement.


inds										=	np.where( Fermi_exclusive_T90 < T90_cut )
Fermi_short_exclusive_name					=	Fermi_exclusive_name[inds]
Fermi_short_exclusive_T90					=	Fermi_exclusive_T90[inds]
Fermi_short_exclusive_T90_error				=	Fermi_exclusive_T90_error[inds]
Fermi_short_exclusive_flux					=	Fermi_exclusive_flux[inds]
Fermi_short_exclusive_flux_error			=	Fermi_exclusive_flux_error[inds]
Fermi_short_exclusive_Epeak_in_MeV			=	Fermi_exclusive_Epeak_in_MeV[inds]
Fermi_short_exclusive_Epeak_in_MeV_error	=	Fermi_exclusive_Epeak_in_MeV_error[inds]
Fermi_short_exclusive_num					=	Fermi_short_exclusive_name.size

print '\n\n\n\n'
print '#### Fermi short exclusive GRBs ####', '\n'
print 'Number of exclusive Fermi GRBs	:	', Fermi_exclusive_num
print 'Out of which, number of short	:	', Fermi_short_exclusive_num, '\n'
Fermi_short_exclusive_name, Fermi_short_exclusive_flux, Fermi_short_exclusive_flux_error, Fermi_short_exclusive_Epeak_in_MeV, Fermi_short_exclusive_Epeak_in_MeV_error, Fermi_short_exclusive_pseudo_redshift, Fermi_short_exclusive_pseudo_redshift_error, Fermi_short_exclusive_Luminosity, Fermi_short_exclusive_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Fermi_exclusive( Fermi_short_exclusive_name, Fermi_short_exclusive_flux, Fermi_short_exclusive_flux_error, Fermi_short_exclusive_Epeak_in_MeV, Fermi_short_exclusive_Epeak_in_MeV_error )
print 'Number of GRBs selected		:	', Fermi_short_exclusive_flux.size
print '\n\n\n\n'


####################################################################################################################################################






####################################################################################################################################################
########	Writing the data.


print '\n\n\n\n'
#	print (  np.where( Fermi_short_exclusive_flux_error			==	0 )[0] - np.where( Fermi_short_exclusive_pseudo_redshift_error	==	0 )[0]  ==  0  ).all()

L_vs_z__known_short	=	Table( [ common_wkr_Fermi_name[inds_common_short], common_wkr_redshift[inds_common_short], common_wkr_Epeak[inds_common_short], common_wkr_Epeak_error[inds_common_short], known_short_Luminosity, common_wkr_Luminosity_error[inds_common_short] ], names = [ 'Fermi name', 'measured z', 'Epeak (meas) [keV]', 'Epeak_error (meas) [keV]', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )
L_vs_z__Fermi_short	=	Table( [ Fermi_short_name, Fermi_short_pseudo_redshift, Fermi_short_pseudo_redshift_error, Fermi_short_Epeak_in_MeV*1e3, Fermi_short_Epeak_in_MeV_error*1e3, Fermi_short_Luminosity, Fermi_short_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (meas) [keV]', 'Epeak_error (meas) [keV]', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )
L_vs_z__Swift_short	=	Table( [ Swift_short_name, Swift_short_pseudo_redshift, Swift_short_pseudo_redshift_error, Swift_short_Epeak_in_MeV*1e3, Swift_short_Epeak_in_MeV_error*1e3, Swift_short_Luminosity, Swift_short_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (siml) [keV]', 'Epeak_error (siml) [keV]', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )
L_vs_z__other_short	=	Table( [ other_short_name, other_short_redshift, other_short_Luminosity, other_short_Luminosity_error ], names = [ 'name', 'measured z', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )
L_vs_z__BATSE_short	=	Table( [ BATSE_short_name, BATSE_short_pseudo_redshift, BATSE_short_pseudo_redshift_error, BATSE_short_Epeak_in_MeV*1e3, BATSE_short_Epeak_in_MeV_error*1e3, BATSE_short_Luminosity, BATSE_short_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (siml) [keV]', 'Epeak_error (siml) [keV]', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )
L_vs_z__FermE_short	=	Table( [ Fermi_short_exclusive_name, Fermi_short_exclusive_pseudo_redshift, Fermi_short_exclusive_pseudo_redshift_error, Fermi_short_exclusive_Epeak_in_MeV*1e3, Fermi_short_exclusive_Epeak_in_MeV_error*1e3, Fermi_short_exclusive_Luminosity, Fermi_short_exclusive_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (siml) [keV]', 'Epeak_error (siml) [keV]', 'Luminosity [erg/s]', 'Luminosity_error [erg/s]' ] )

#~ ascii.write( L_vs_z__known_short, './../tables/L_vs_z__known_short.txt', format = 'fixed_width', overwrite = True )
#~ ascii.write( L_vs_z__Fermi_short, './../tables/L_vs_z__Fermi_short.txt', format = 'fixed_width', overwrite = True )
#~ ascii.write( L_vs_z__Swift_short, './../tables/L_vs_z__Swift_short.txt', format = 'fixed_width', overwrite = True )
#~ ascii.write( L_vs_z__other_short, './../tables/L_vs_z__other_short.txt', format = 'fixed_width', overwrite = True )
#~ ascii.write( L_vs_z__BATSE_short, './../tables/L_vs_z__BATSE_short.txt', format = 'fixed_width', overwrite = True )
#~ ascii.write( L_vs_z__FermE_short, './../tables/L_vs_z__FermE_short.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################
