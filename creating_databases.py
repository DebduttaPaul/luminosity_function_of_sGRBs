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



####################################################################################################################################################






####################################################################################################################################################
########	Reading the data.



Fermi_GRBs_table			=	ascii.read( './../tables/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_name					=	Fermi_GRBs_table['Fermi name'].data
Fermi_Ttime					=	Fermi_GRBs_table['GBM Trigger-time'].data
Fermi_T90					=	Fermi_GRBs_table['GBM T90'].data
Fermi_T90_error				=	Fermi_GRBs_table['GBM T90_error'].data
Fermi_flux					=	Fermi_GRBs_table['GBM flux'].data
Fermi_flux_error			=	Fermi_GRBs_table['GBM flux_error'].data
Fermi_fluence				=	Fermi_GRBs_table['GBM fluence'].data
Fermi_fluence_error			=	Fermi_GRBs_table['GBM fluence_error'].data
Fermi_Epeak      			=	Fermi_GRBs_table['Epeak'].data
Fermi_Epeak_error			=	Fermi_GRBs_table['Epeak_error'].data
Fermi_alpha					=	Fermi_GRBs_table['alpha'].data
Fermi_alpha_error			=	Fermi_GRBs_table['alpha_error'].data
Fermi_beta					=	Fermi_GRBs_table['beta'].data
Fermi_beta_error			=	Fermi_GRBs_table['beta_error'].data
Fermi_num					=	Fermi_name.size



Swift_all_GRBs_table		=	ascii.read( './../tables/Swift_GRBs--all.txt', format = 'fixed_width' )
Swift_all_name				=	Swift_all_GRBs_table['Swift name'].data
Swift_all_Ttimes			=	Swift_all_GRBs_table['BAT Trigger-time'].data
Swift_all_T90				=	Swift_all_GRBs_table['BAT T90'].data
Swift_all_flux				=	Swift_all_GRBs_table['BAT Phoflux'].data
Swift_all_flux_error		=	Swift_all_GRBs_table['BAT Phoflux_error'].data
Swift_all_fluence			=	Swift_all_GRBs_table['BAT fluence'].data
Swift_all_fluence_error		=	Swift_all_GRBs_table['BAT fluence_error'].data
Swift_all_num				=	Swift_all_name.size

Swift_wkr_GRBs_table		=	ascii.read( './../tables/Swift_GRBs--wkr.txt', format = 'fixed_width' )
Swift_wkr_name				=	Swift_wkr_GRBs_table['Swift name'].data
Swift_wkr_Ttimes			=	Swift_wkr_GRBs_table['BAT Trigger-time'].data
Swift_wkr_redhsift			=	Swift_wkr_GRBs_table['redshift'].data
Swift_wkr_T90				=	Swift_wkr_GRBs_table['BAT T90'].data
Swift_wkr_flux				=	Swift_wkr_GRBs_table['BAT Phoflux'].data
Swift_wkr_flux_error		=	Swift_wkr_GRBs_table['BAT Phoflux_error'].data
Swift_wkr_fluence			=	Swift_wkr_GRBs_table['BAT fluence'].data
Swift_wkr_fluence_error		=	Swift_wkr_GRBs_table['BAT fluence_error'].data
Swift_wkr_num				=	Swift_wkr_name.size
Swift_wkr_num				=	Swift_wkr_name.size



common_all_GRBs_table			=	ascii.read( './../tables/common_GRBs--all.txt', format = 'fixed_width' )
common_all_ID					=	common_all_GRBs_table['common ID'].data
common_all_Swift_name			=	common_all_GRBs_table['Swift name'].data
common_all_Fermi_name			=	common_all_GRBs_table['Fermi name'].data
common_all_Swift_T90			=	common_all_GRBs_table['BAT T90'].data
common_all_Fermi_T90			=	common_all_GRBs_table['GBM T90'].data
common_all_Fermi_T90_error		=	common_all_GRBs_table['GBM T90_error'].data
common_all_Fermi_flux			=	common_all_GRBs_table['GBM flux'].data
common_all_Fermi_flux_error		=	common_all_GRBs_table['GBM flux_error'].data
common_all_Fermi_fluence		=	common_all_GRBs_table['GBM fluence'].data
common_all_Fermi_fluence_error	=	common_all_GRBs_table['GBM fluence_error'].data
common_all_Epeak				=	common_all_GRBs_table['Epeak'].data				#	in keV.
common_all_Epeak_error			=	common_all_GRBs_table['Epeak_error'].data		#	in keV.
common_all_alpha				=	common_all_GRBs_table['alpha'].data
common_all_alpha_error			=	common_all_GRBs_table['alpha_error'].data
common_all_beta					=	common_all_GRBs_table['beta'].data
common_all_beta_error			=	common_all_GRBs_table['beta_error'].data
common_all_num					=	common_all_ID.size

common_wkr_GRBs_table			=	ascii.read( './../tables/common_GRBs--wkr.txt', format = 'fixed_width' )
common_wkr_ID					=	common_wkr_GRBs_table['common ID'].data
common_wkr_Swift_name			=	common_wkr_GRBs_table['Swift name'].data
common_wkr_Fermi_name			=	common_wkr_GRBs_table['Fermi name'].data
common_wkr_Fermi_Ttime			=	common_wkr_GRBs_table['GBM Trigger-time'].data	#	in hours.
common_wkr_Swift_T90			=	common_wkr_GRBs_table['BAT T90'].data
common_wkr_Fermi_T90			=	common_wkr_GRBs_table['GBM T90'].data
common_wkr_Fermi_T90_error		=	common_wkr_GRBs_table['GBM T90_error'].data
common_wkr_redshift				=	common_wkr_GRBs_table['redshift'].data
common_wkr_Fermi_flux			=	common_wkr_GRBs_table['GBM flux'].data
common_wkr_Fermi_flux_error		=	common_wkr_GRBs_table['GBM flux_error'].data
common_wkr_Fermi_fluence		=	common_wkr_GRBs_table['GBM fluence'].data
common_wkr_Fermi_fluence_error	=	common_wkr_GRBs_table['GBM fluence_error'].data
common_wkr_Epeak				=	common_wkr_GRBs_table['Epeak'].data				#	in keV.
common_wkr_Epeak_error			=	common_wkr_GRBs_table['Epeak_error'].data		#	in keV.
common_wkr_alpha				=	common_wkr_GRBs_table['alpha'].data
common_wkr_alpha_error			=	common_wkr_GRBs_table['alpha_error'].data
common_wkr_beta					=	common_wkr_GRBs_table['beta'].data
common_wkr_beta_error			=	common_wkr_GRBs_table['beta_error'].data
common_wkr_Luminosity			=	common_wkr_GRBs_table['Luminosity'].data
common_wkr_Luminosity_error		=	common_wkr_GRBs_table['Luminosity_error'].data
common_wkr_num					=	common_wkr_ID.size



BATSE_GRBs_table			=	ascii.read( './../tables/BATSE_GRBs--measured.txt', format = 'fixed_width' )
BATSE_name					=	BATSE_GRBs_table['name'].data
BATSE_Ttime					=	BATSE_GRBs_table['T-time'].data
BATSE_T90					=	BATSE_GRBs_table['T90'].data
BATSE_T90_error				=	BATSE_GRBs_table['T90_error'].data
BATSE_flux					=	BATSE_GRBs_table['flux'].data
BATSE_flux_error			=	BATSE_GRBs_table['flux_error'].data
BATSE_fluence1				=	BATSE_GRBs_table['fluence_1'].data
BATSE_fluence1_error		=	BATSE_GRBs_table['fluence_1_error'].data
BATSE_fluence2				=	BATSE_GRBs_table['fluence_2'].data
BATSE_fluence2_error		=	BATSE_GRBs_table['fluence_2_error'].data
BATSE_fluence3				=	BATSE_GRBs_table['fluence_3'].data
BATSE_fluence3_error		=	BATSE_GRBs_table['fluence_3_error'].data
BATSE_fluence4				=	BATSE_GRBs_table['fluence_4'].data
BATSE_fluence4_error		=	BATSE_GRBs_table['fluence_4_error'].data
BATSE_num					=	BATSE_name.size
print 'Number of BATSE GRBs	:	' , BATSE_num
#	inds						=	np.where( BATSE_flux > BATSE_sensitivity )[0]
inds						=	np.where( BATSE_flux != 0 )[0]
BATSE_name					=	BATSE_name[inds]
BATSE_Ttime					=	BATSE_Ttime[inds]
BATSE_T90					=	BATSE_T90[inds]
BATSE_T90_error				=	BATSE_T90_error[inds]
BATSE_flux					=	BATSE_flux[inds]
BATSE_flux_error			=	BATSE_flux_error[inds]
BATSE_fluence1				=	BATSE_fluence1[inds]
BATSE_fluence1_error		=	BATSE_fluence1_error[inds]
BATSE_fluence2				=	BATSE_fluence2[inds]
BATSE_fluence2_error		=	BATSE_fluence2_error[inds]
BATSE_fluence3				=	BATSE_fluence3[inds]
BATSE_fluence3_error		=	BATSE_fluence3_error[inds]
BATSE_fluence4				=	BATSE_fluence4[inds]
BATSE_fluence4_error		=	BATSE_fluence4_error[inds]
BATSE_num					=	BATSE_name.size
print 'Number of BATSE GRBs	:	' , BATSE_num



Fermi_short_exclusive_GRBs_table	=	ascii.read( './../tables/Fermi_GRBs--without_spectral_parameters--short.txt', format = 'fixed_width' )
Fermi_short_exclusive_name			=	Fermi_short_exclusive_GRBs_table['name'].data
Fermi_short_exclusive_Ttime			=	Fermi_short_exclusive_GRBs_table['Ttime'].data
Fermi_short_exclusive_T90			=	Fermi_short_exclusive_GRBs_table['T90'].data
Fermi_short_exclusive_T90_error		=	Fermi_short_exclusive_GRBs_table['T90_error'].data
Fermi_short_exclusive_flux			=	Fermi_short_exclusive_GRBs_table['flux [erg.cm^{-2}.s^{-1}]'].data
Fermi_short_exclusive_flux_error	=	Fermi_short_exclusive_GRBs_table['flux_error [erg.cm^{-2}.s^{-1}]'].data
Fermi_short_exclusive_fluence		=	Fermi_short_exclusive_GRBs_table['fluence [erg.cm^{-2}]'].data
Fermi_short_exclusive_fluence_error	=	Fermi_short_exclusive_GRBs_table['fluence_error [erg.cm^{-2}]'].data
Fermi_short_exclusive_num			=	Fermi_short_exclusive_name.size

print '\n\n'
print Fermi_short_exclusive_flux.min(), Fermi_short_exclusive_flux.max()
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





Fermi_short_name			=	Fermi_name[inds_short_in_Fermi_without_redshift]
Fermi_short_Ttime			=	Fermi_Ttime[inds_short_in_Fermi_without_redshift]
Fermi_short_T90				=	Fermi_T90[inds_short_in_Fermi_without_redshift]
Fermi_short_T90_error		=	Fermi_T90_error[inds_short_in_Fermi_without_redshift]
Fermi_short_flux			=	Fermi_flux[inds_short_in_Fermi_without_redshift]
Fermi_short_flux_error		=	Fermi_flux_error[inds_short_in_Fermi_without_redshift]
Fermi_short_fluence			=	Fermi_fluence[inds_short_in_Fermi_without_redshift]
Fermi_short_fluence_error	=	Fermi_fluence_error[inds_short_in_Fermi_without_redshift]



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
Swift_Ttime			=	Swift_all_Ttimes[inds_exclusively_Swift_GRBs_without_redshift]
Swift_T90			=	Swift_all_T90[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux			=	Swift_all_flux[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux_error	=	Swift_all_flux_error[inds_exclusively_Swift_GRBs_without_redshift]
Swift_fluence		=	Swift_all_fluence[inds_exclusively_Swift_GRBs_without_redshift]
Swift_fluence_error	=	Swift_all_fluence_error[inds_exclusively_Swift_GRBs_without_redshift]





inds_long_in_Swift	=	np.where( Swift_T90 >= T90_cut )[0]
inds_short_in_Swift	=	np.delete( np.arange(Swift_num), inds_long_in_Swift )

Swift_long_num					=	inds_long_in_Swift.size

Swift_short_name				=	Swift_name[inds_short_in_Swift]
Swift_short_Ttime				=	Swift_Ttime[inds_short_in_Swift]
Swift_short_T90					=	Swift_T90[inds_short_in_Swift]
Swift_short_flux				=	Swift_flux[inds_short_in_Swift]
Swift_short_flux_error			=	np.round( Swift_flux_error[inds_short_in_Swift]   , 3)
Swift_short_fluence				=	Swift_fluence[inds_short_in_Swift]
Swift_short_fluence_error		=	np.round( Swift_fluence_error[inds_short_in_Swift], 3)
Swift_short_num					=	Swift_short_name.size

print 'Out of which, # of long  GRBs	:	', Swift_long_num
print '                   short GRBs	:	', Swift_short_num


print '\n\n\n\n'
print '#### Swift short GRBs ####', '\n'
print 'Number of GRBs put in		:	', Swift_short_num, '\n'
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
other_Swift_Ttime			=	Swift_wkr_Ttimes[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_redshift		=	Swift_wkr_redhsift[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_T90				=	Swift_wkr_T90[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux			=	Swift_wkr_flux[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux_error		=	Swift_wkr_flux_error[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_fluence			=	Swift_wkr_fluence[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_fluence_error	=	Swift_wkr_fluence_error[inds_exclusively_Swift_GRBs_with_redshift]



inds_other_long		=	np.where( other_Swift_T90 >= T90_cut )[0]
inds_other_short	=	np.delete( np.arange(other_num), inds_other_long )

other_long_num			=	inds_other_long.size

other_short_name			=	other_Swift_name[inds_other_short]
other_short_Ttime			=	other_Swift_Ttime[inds_other_short]
other_short_T90				=	other_Swift_T90[inds_other_short]
other_short_redshift		=	other_Swift_redshift[inds_other_short]
other_short_flux			=	other_Swift_flux[inds_other_short]
other_short_flux_error		=	np.round( other_Swift_flux_error[inds_other_short]    , 3 )
other_short_fluence			=	other_Swift_fluence[inds_other_short]
other_short_fluence_error	=	np.round( other_Swift_fluence_error[inds_other_short] , 3 )
other_short_num				=	inds_other_short.size


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


inds						=	np.where( BATSE_T90 < T90_cut )
BATSE_short_name			=	BATSE_name[inds]
BATSE_short_Ttime			=	BATSE_Ttime[inds]
BATSE_short_T90				=	BATSE_T90[inds]
BATSE_short_T90_error		=	BATSE_T90_error[inds]
BATSE_short_flux			=	BATSE_flux[inds]
BATSE_short_flux_error		=	BATSE_flux_error[inds]
BATSE_short_fluence1		=	BATSE_fluence1[inds]
BATSE_short_fluence1_error	=	BATSE_fluence1_error[inds]
BATSE_short_fluence2		=	BATSE_fluence2[inds]
BATSE_short_fluence2_error	=	BATSE_fluence2_error[inds]
BATSE_short_fluence3		=	BATSE_fluence3[inds]
BATSE_short_fluence3_error	=	BATSE_fluence3_error[inds]
BATSE_short_fluence4		=	BATSE_fluence4[inds]
BATSE_short_fluence4_error	=	BATSE_fluence4_error[inds]
BATSE_short_num				=	BATSE_name.size

print '\n\n\n\n'
print '#### BATSE short GRBs ####', '\n'
print 'Number of GRBs put in		:	', BATSE_short_flux.size, '\n'
print '\n\n\n\n'


####################################################################################################################################################







####################################################################################################################################################
########	For the exclsuive Fermi GRBs (i.e. not common to Swift) without spectral parameter measurement.


print '\n\n\n\n'
print '#### Fermi short exclusive GRBs ####', '\n'
print 'Number of exclusive Fermi short GRBs	:	', Fermi_short_exclusive_num

Fermi_short_name			=	np.concatenate( [ Fermi_short_name, Fermi_short_exclusive_name] )
Fermi_short_Ttime			=	np.concatenate( [ Fermi_short_Ttime, Fermi_short_exclusive_Ttime] )
Fermi_short_T90				=	np.concatenate( [ Fermi_short_T90, Fermi_short_exclusive_T90] )
Fermi_short_T90_error		=	np.concatenate( [ Fermi_short_T90_error, Fermi_short_exclusive_T90_error] )
Fermi_short_flux			=	np.concatenate( [ Fermi_short_flux, Fermi_short_exclusive_flux] )
Fermi_short_flux_error		=	np.concatenate( [ Fermi_short_flux_error, Fermi_short_exclusive_flux_error] )
Fermi_short_fluence			=	np.concatenate( [ Fermi_short_fluence_error, Fermi_short_exclusive_fluence] )
Fermi_short_fluence_error	=	np.concatenate( [ Fermi_short_fluence_error, Fermi_short_exclusive_fluence_error] )


print '\n\n\n\n'


####################################################################################################################################################






####################################################################################################################################################
########	Writing the data.


print '\n\n\n\n'
#	print (  np.where( Fermi_short_exclusive_flux_error			==	0 )[0] - np.where( Fermi_short_exclusive_pseudo_redshift_error	==	0 )[0]  ==  0  ).all()

database_short__known	=	Table( [ common_wkr_Swift_name[inds_common_short], common_wkr_Fermi_name[inds_common_short], common_wkr_Fermi_Ttime[inds_common_short], common_wkr_Swift_T90[inds_common_short], common_wkr_Fermi_T90[inds_common_short], common_wkr_Fermi_T90_error[inds_common_short], common_wkr_Fermi_flux[inds_common_short], common_wkr_Fermi_flux_error[inds_common_short],  common_wkr_Fermi_fluence[inds_common_short], common_wkr_Fermi_fluence_error[inds_common_short], common_wkr_redshift[inds_common_short] ], names = [ 'Swift name', 'Fermi name', 'T-time [hrs]', 'BAT T90 [s]', 'GBM T90 [s]', 'GBM T90_error [s]', 'GBM flux', 'GBM flux_error', 'GBM fluence', 'GBM fluence_error', 'measured z' ] )
database_short__Fermi	=	Table( [ Fermi_short_name, Fermi_short_Ttime, Fermi_short_T90, Fermi_short_T90_error, Fermi_short_flux, Fermi_short_flux_error, Fermi_short_fluence, Fermi_short_fluence_error ], names = [ 'name', 'T-time [hrs]', 'T90 [s]', 'T90_error [s]', 'flux', 'flux_error', 'fluence', 'fluence_error' ] )
database_short__Swift	=	Table( [ Swift_short_name, Swift_short_Ttime, Swift_short_T90, Swift_short_flux, Swift_short_flux_error, Swift_short_fluence, Swift_short_fluence_error ]                       , names = [ 'name', 'T-time [hrs]', 'T90 [s]', 'flux', 'flux_error', 'fluence', 'fluence_error' ] )
database_short__other	=	Table( [ other_short_name, other_short_Ttime, other_short_T90, other_short_flux, other_short_flux_error, other_short_fluence, other_short_fluence_error, other_short_redshift ] , names = [ 'name', 'T-time [hrs]', 'T90 [s]', 'flux', 'flux_error', 'fluence', 'fluence_error', 'measured z' ] )
database_short__BATSE	=	Table( [ BATSE_short_name, BATSE_short_Ttime, BATSE_short_T90, BATSE_short_T90_error, BATSE_short_flux, BATSE_short_flux_error, BATSE_short_fluence1, BATSE_short_fluence1_error, BATSE_short_fluence2, BATSE_short_fluence2_error, BATSE_short_fluence3, BATSE_short_fluence3_error, BATSE_short_fluence4, BATSE_short_fluence4_error ], names = [ 'name', 'T-time', 'T90 [s]', 'T90_error [s]', 'flux', 'flux_error', 'fluence_1', 'fluence_1_error', 'fluence_2', 'fluence_2_error', 'fluence_3', 'fluence_3_error', 'fluence_4', 'fluence_4_error' ] )

ascii.write( database_short__known, './../tables/database_short__known.txt', format = 'fixed_width', overwrite = True )
ascii.write( database_short__Fermi, './../tables/database_short__Fermi.txt', format = 'fixed_width', overwrite = True )
ascii.write( database_short__Swift, './../tables/database_short__Swift.txt', format = 'fixed_width', overwrite = True )
ascii.write( database_short__other, './../tables/database_short__other.txt', format = 'fixed_width', overwrite = True )
ascii.write( database_short__BATSE, './../tables/database_short__BATSE.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################
