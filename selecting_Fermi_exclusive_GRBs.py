from __future__ import division
from astropy.io import ascii
from astropy.table import Table
import debduttaS_functions as mf
import specific_functions as sf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	10	# The size of markers in scatter plots.

P	=	np.pi		# Dear old pi!

T90_cut		=	2		#	in sec.


####################################################################################################################################################





####################################################################################################################################################


def choose( bigger, smaller ):
	
	
	indices = []
	
	for i, s in enumerate( smaller ):
		ind	=	np.where(bigger == s)[0][0]		# the index is of the bigger array.
		indices.append( ind )
	
	
	return np.array(indices)


def convert( mother, mins, secs ):
	return ( mother*(60**2) + mins*60 + secs ) / (60**2)	# in hours


####################################################################################################################################################






####################################################################################################################################################


Swift_all_GRBs_table	=	ascii.read( './../tables/Swift_GRBs--all.txt', format = 'fixed_width' )
Swift_all_name			=	Swift_all_GRBs_table['Swift name'].data
Swift_all_ID			=	Swift_all_GRBs_table['Swift ID'].data
Swift_all_Ttimes		=	Swift_all_GRBs_table['BAT Trigger-time'].data
Swift_all_RA			=	Swift_all_GRBs_table['BAT RA'].data
Swift_all_Dec			=	Swift_all_GRBs_table['BAT Dec'].data
Swift_all_error_radius	=	Swift_all_GRBs_table['BAT Error-radius'].data
Swift_all_num			=	Swift_all_name.size
print 'Total number of Swift GRBs	:	', Swift_all_num, '\n'


####################################################################################################################################################






####################################################################################################################################################



Fermi_all_data			=	ascii.read( './../data/Fermi--all_GRBs.txt', format = 'fixed_width' )
Fermi_all_GRBs_name		=	Fermi_all_data['name'].data
Fermi_all_Ttimes		=	Fermi_all_data['trigger_time'].data
Fermi_all_RAs			=	Fermi_all_data['ra'].data						#	in  hr,min,sec.
Fermi_all_Decs			=	Fermi_all_data['dec'].data						#	in deg,min,sec.
Fermi_all_error_radii	=	Fermi_all_data['error_radius'].data				#	in degree.
Fermi_all_T90			=	Fermi_all_data['t90'].data						#	in sec.
Fermi_all_T90_error		=	Fermi_all_data['t90_error'].data				#	in sec.
Fermi_all_flux			=	Fermi_all_data['pflx_band_ergflux'].data		#	in erg.cm^{-2}.s^{-1}.
Fermi_all_flux_error	=	Fermi_all_data['pflx_band_ergflux_error'].data	#	same as above.
Fermi_all_fluence		=	Fermi_all_data['pflx_band_ergflnc'].data		#	in erg.cm^{-2}.
Fermi_all_fluence_error	=	Fermi_all_data['pflx_band_ergflnc_error'].data	#	same as above.
Fermi_all_num			=	Fermi_all_GRBs_name.size
print 'Total number of Fermi GRBs	:	', Fermi_all_num
inds					=	np.where( np.ma.getmask( Fermi_all_flux ) == False )
Fermi_all_GRBs_name		=	Fermi_all_GRBs_name[inds]
Fermi_all_Ttimes		=	Fermi_all_Ttimes[inds]
Fermi_all_RAs			=	Fermi_all_RAs[inds]
Fermi_all_Decs			=	Fermi_all_Decs[inds]
Fermi_all_error_radii	=	Fermi_all_error_radii[inds]
Fermi_all_T90			=	Fermi_all_T90[inds]
Fermi_all_T90_error		=	Fermi_all_T90_error[inds]
Fermi_all_flux			=	Fermi_all_flux[inds]
Fermi_all_flux_error	=	Fermi_all_flux_error[inds]
Fermi_all_fluence		=	Fermi_all_fluence[inds]
Fermi_all_fluence_error	=	Fermi_all_fluence_error[inds]
Fermi_all_num			=	Fermi_all_GRBs_name.size
print 'With      flux-measurements	:	', Fermi_all_num


Fermi_wsp_table			=	ascii.read( './../tables/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_wsp_name			=	Fermi_wsp_table['Fermi name'].data
Fermi_wsp_T90			=	Fermi_wsp_table['GBM T90'].data
Fermi_wsp_Epeak      	=	Fermi_wsp_table['Epeak'].data
Fermi_wsp_num			=	Fermi_wsp_name.size
print 'With    spectral parameters	:	', Fermi_wsp_num
inds_Fermi_wsp_short	=	np.where(Fermi_wsp_T90 < T90_cut)[0]	;	inds_Fermi_wsp_long	=	np.delete( np.arange(Fermi_wsp_num), inds_Fermi_wsp_short )
print '	...subset:	short	:	', inds_Fermi_wsp_short.size

inds					=	choose( Fermi_all_GRBs_name, Fermi_wsp_name )
inds					=	np.delete( np.arange(Fermi_all_num), inds )
Fermi_rest_name			=	Fermi_all_GRBs_name[inds]
Fermi_rest_Ttimes		=	Fermi_all_Ttimes[inds]
Fermi_rest_RAs			=	Fermi_all_RAs[inds]
Fermi_rest_Decs			=	Fermi_all_Decs[inds]
Fermi_rest_error_radii	=	Fermi_all_error_radii[inds]
Fermi_rest_T90			=	Fermi_all_T90[inds]
Fermi_rest_T90_error	=	Fermi_all_T90_error[inds]
Fermi_rest_flux			=	Fermi_all_flux[inds]
Fermi_rest_flux_error	=	Fermi_all_flux_error[inds]
Fermi_rest_fluence		=	Fermi_all_fluence[inds]
Fermi_rest_fluence_error=	Fermi_all_fluence_error[inds]
Fermi_rest_num			=	Fermi_rest_name.size
print 'Without spectral parameters	:	', Fermi_rest_num


####################################################################################################################################################




####################################################################################################################################################


Fermi_rest_ID	=	np.zeros( Fermi_rest_num )
Fermi_rest_Tt	=	Fermi_rest_ID.copy()
Fermi_rest_RA	=	Fermi_rest_ID.copy()
Fermi_rest_Dec	=	Fermi_rest_ID.copy()
for j, name in enumerate(Fermi_rest_name):
	Ttime	=	Fermi_rest_Ttimes[j]
	RA		=	Fermi_rest_RAs[j]
	Dec		=	Fermi_rest_Decs[j]
	
	Fermi_rest_ID[j]	=	name[3:9]
	
	Fermi_rest_Tt[j]	=	convert( float(Ttime[11:13]), float(Ttime[14:16]), float(Ttime[17:23]) )
	
	Fermi_rest_RA[j]	=	convert( float(RA[0: 2]), float(RA[3: 5]), float(RA[6:10]) )
	
	sign		=	Dec[0:1]
	decimal		=	convert( float(Dec[1:3]), float(Dec[4:6]), float(Dec[7:9]) )
	if sign == '-':	decimal = decimal * (-1)
	Fermi_rest_Dec[j]	=	decimal

Fermi_rest_ID	=	Fermi_rest_ID.astype(int)
Fermi_rest_Tt	=	np.round( Fermi_rest_Tt , 3 )
Fermi_rest_RA	=	np.round( Fermi_rest_RA , 3 )
Fermi_rest_Dec	=	np.round( Fermi_rest_Dec, 3 )


indices_exclusive	=	np.array([])
for j, ID in enumerate(Fermi_rest_ID):
	
	ind	=	np.where( Swift_all_ID == ID )[0]
	if ind.size != 0:
		
		diff_time	=	np.abs( Fermi_rest_Tt[j]  - Swift_all_Ttimes[ind] )
		diff_RA		=	np.abs( Fermi_rest_RA[j]  - Swift_all_RA[ind]     )
		diff_Dec	=	np.abs( Fermi_rest_Dec[j] - Swift_all_Dec[ind]    )
		check		=	np.where( (diff_time < 10/60) & (diff_RA < 10) & (diff_Dec < 10) )[0]			#	experimentally set at the convergent value for diff_time, doesn't change beyond 5 mins, all the way up to 10 mins; similarly for RA and Dec, roughly 10 degree by 10 degree (Fermi errors).
		
		if check.size == 0:	indices_exclusive	=	np.append( indices_exclusive, j )
	
	else:	indices_exclusive	=	np.append( indices_exclusive, j )

indices_exclusive	=	indices_exclusive.astype(int)

#	print Fermi_rest_name[indices_exclusive]

Fermi_excl_num	=	indices_exclusive.size
print 'Those exclusive to Fermi	:	', Fermi_excl_num


Fermi_exclusive_name				=	Fermi_rest_name[indices_exclusive]
Fermi_exclusive_ID					=	Fermi_rest_ID[indices_exclusive]
Fermi_exclusive_Ttimes				=	Fermi_rest_Ttimes[indices_exclusive]
Fermi_exclusive_T90					=	Fermi_rest_T90[indices_exclusive]
Fermi_exclusive_T90_error			=	Fermi_rest_T90_error[indices_exclusive]
Fermi_exclusive_flux				=	Fermi_rest_flux[indices_exclusive]
Fermi_exclusive_flux_error			=	Fermi_rest_flux_error[indices_exclusive]
Fermi_exclusive_fluence				=	Fermi_rest_fluence[indices_exclusive]
Fermi_exclusive_fluence_error		=	Fermi_rest_fluence_error[indices_exclusive]
Fermi_exclusive_num					=	Fermi_exclusive_name.size


inds_Fermi_excl_short	=	np.where(Fermi_exclusive_T90 < T90_cut)[0]	;	inds_Fermi_excl_long	=	np.delete( np.arange(Fermi_excl_num), inds_Fermi_excl_short )
Fermi_excl_short_name			=	Fermi_exclusive_name[inds_Fermi_excl_short]
Fermi_excl_short_Ttime			=	Fermi_exclusive_Ttimes[inds_Fermi_excl_short]
Fermi_excl_short_T90			=	Fermi_exclusive_T90[inds_Fermi_excl_short]
Fermi_excl_short_T90_error		=	Fermi_exclusive_T90_error[inds_Fermi_excl_short]
Fermi_excl_short_flux			=	Fermi_exclusive_flux[inds_Fermi_excl_short]
Fermi_excl_short_flux_error		=	Fermi_exclusive_flux_error[inds_Fermi_excl_short]
Fermi_excl_short_fluence		=	Fermi_exclusive_fluence[inds_Fermi_excl_short]
Fermi_excl_short_fluence_error	=	Fermi_exclusive_fluence_error[inds_Fermi_excl_short]
Fermi_excl_short_num			=	Fermi_excl_short_name.size
print '	...subset:	short	:	', Fermi_excl_short_num
Fermi_excl_long_name			=	Fermi_exclusive_name[inds_Fermi_excl_long]
Fermi_excl_long_Ttime			=	Fermi_exclusive_Ttimes[inds_Fermi_excl_long]
Fermi_excl_long_T90				=	Fermi_exclusive_T90[inds_Fermi_excl_long]
Fermi_excl_long_T90_error		=	Fermi_exclusive_T90_error[inds_Fermi_excl_long]
Fermi_excl_long_flux			=	Fermi_exclusive_flux[inds_Fermi_excl_long]
Fermi_excl_long_flux_error		=	Fermi_exclusive_flux_error[inds_Fermi_excl_long]
Fermi_excl_long_fluence			=	Fermi_exclusive_fluence[inds_Fermi_excl_long]
Fermi_excl_long_fluence_error	=	Fermi_exclusive_fluence_error[inds_Fermi_excl_long]
Fermi_excl_long_num				=	Fermi_excl_long_name.size
print '	...subset:	long	:	', Fermi_excl_long_num


####################################################################################################################################################





####################################################################################################################################################


hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_wsp_Epeak[inds_Fermi_wsp_short]), 0.50, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]
print '\n\n'
print 'short Epeak mean [keV]: ', round( 10**f0, 3 )

Fermi_excl_short_Epeak_in_keV		=	np.random.normal( f0, f1, Fermi_excl_short_num )
Fermi_excl_short_Epeak_in_keV		=	10**Fermi_excl_short_Epeak_in_keV			#	in keV
Fermi_excl_short_Epeak_in_MeV		=	1e-3 * Fermi_excl_short_Epeak_in_keV		#	in MeV
Fermi_excl_short_Epeak_in_MeV_error	=	np.zeros( Fermi_excl_short_Epeak_in_MeV.size )

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_excl_short_Epeak_in_keV), 0.50, 1, 4 )	;	fx	=	hist[0]	;	fy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
plt.step( hx, hy, where = 'mid', color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
plt.bar( fx, fy * (hy.max()/fy.max()), color = 'w', width = 0.1, edgecolor = 'k', hatch = '//', label = r'$ Fermi \rm{ , \; simulated } $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/Fermi_exclusive--Ep_distribution--short.png' )
plt.clf()
plt.close()



hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_wsp_Epeak[inds_Fermi_wsp_long ]), 0.25, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]
print ' long Epeak mean [keV]: ', round( 10**f0, 3 )

Fermi_excl_long_Epeak_in_keV		=	np.random.normal( f0, f1, Fermi_excl_long_num )
Fermi_excl_long_Epeak_in_keV		=	10**Fermi_excl_long_Epeak_in_keV			#	in keV
Fermi_excl_long_Epeak_in_MeV		=	1e-3 * Fermi_excl_long_Epeak_in_keV		#	in MeV
Fermi_excl_long_Epeak_in_MeV_error	=	np.zeros( Fermi_excl_long_Epeak_in_MeV.size )

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_excl_long_Epeak_in_keV), 0.25, 1, 4 )	;	fx	=	hist[0]	;	fy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
plt.step( hx, hy, where = 'mid', color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
plt.bar( fx, fy * (hy.max()/fy.max()), color = 'w', width = 0.1, edgecolor = 'k', hatch = '//', label = r'$ Fermi \rm{ , \; simulated } $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/Fermi_exclusive--Ep_distribution--long.png' )
plt.clf()
plt.close()


####################################################################################################################################################




####################################################################################################################################################



Fermi_excl_short_table	=	Table( [ Fermi_excl_short_name, Fermi_excl_short_Ttime, Fermi_excl_short_T90, Fermi_excl_short_T90_error, Fermi_excl_short_flux, Fermi_excl_short_flux_error, Fermi_excl_short_fluence, Fermi_excl_short_fluence_error, Fermi_excl_short_Epeak_in_MeV, Fermi_excl_short_Epeak_in_MeV_error ], 
									names = [ 'name', 'Ttime', 'T90', 'T90_error', 'flux [erg.cm^{-2}.s^{-1}]', 'flux_error [erg.cm^{-2}.s^{-1}]', 'fluence [erg.cm^{-2}]', 'fluence_error [erg.cm^{-2}]', 'Epeak (siml) [MeV]', 'Epeak_error (siml) [MeV]' ] )
ascii.write( Fermi_excl_short_table, './../tables/Fermi_GRBs--without_spectral_parameters--short.txt', format = 'fixed_width', overwrite = True )

Fermi_excl_long_table	=	Table( [ Fermi_excl_long_name , Fermi_excl_long_Ttime , Fermi_excl_long_T90 , Fermi_excl_long_T90_error , Fermi_excl_long_flux , Fermi_excl_long_flux_error , Fermi_excl_long_fluence , Fermi_excl_long_fluence_error , Fermi_excl_long_Epeak_in_MeV , Fermi_excl_long_Epeak_in_MeV_error  ], 
									names = [ 'name', 'Ttime', 'T90', 'T90_error', 'flux [erg.cm^{-2}.s^{-1}]', 'flux_error [erg.cm^{-2}.s^{-1}]', 'fluence [erg.cm^{-2}]', 'fluence_error [erg.cm^{-2}]', 'Epeak (siml) [MeV]', 'Epeak_error (siml) [MeV]' ] )
ascii.write( Fermi_excl_long_table, './../tables/Fermi_GRBs--without_spectral_parameters--long.txt', format = 'fixed_width', overwrite = True )



####################################################################################################################################################

