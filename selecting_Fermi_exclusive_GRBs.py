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

L_norm		=	1e52	#	in ergs.s^{-1}.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6020 * 1e-9


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


Swift_all_GRBs_table	=	ascii.read( './../data/Swift_GRBs--all.txt', format = 'fixed_width' )
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


Fermi_wsp_table			=	ascii.read( './../data/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_wsp_name			=	Fermi_wsp_table['Fermi name'].data
Fermi_wsp_Epeak      	=	Fermi_wsp_table['Epeak'].data
Fermi_wsp_num			=	Fermi_wsp_name.size
print 'With    spectral parameters	:	', Fermi_wsp_num


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


hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_wsp_Epeak), 0.125, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]

Fermi_exclusive_Epeak_in_keV		=	np.random.normal( f0, f1, Fermi_wsp_num )
Fermi_exclusive_Epeak_in_keV		=	10**Fermi_exclusive_Epeak_in_keV		#	in keV
Fermi_exclusive_Epeak_in_MeV		=	1e-3 * Fermi_exclusive_Epeak_in_keV		#	in MeV
Fermi_exclusive_Epeak_in_MeV_error	=	np.zeros( Fermi_exclusive_Epeak_in_MeV.size )

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_exclusive_Epeak_in_keV), 0.125, 1, 4 )	;	fx	=	hist[0]	;	fy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
plt.bar( fx, fy * (hy.max()/fy.max()), color = 'w', width = 0.1, edgecolor = 'k', hatch = '//', label = r'$ Fermi \rm{ , \; simulated } $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/Fermi--Ep_distribution--simulated_1.png' )
plt.clf()
plt.close()

plt.title( r'$ Fermi $', fontsize = size_font )
plt.hist( Fermi_exclusive_Epeak_in_keV, bins = np.logspace(1, 4, 20) )
plt.gca().set_xscale('log')
plt.xlabel( r'$ E_p \; \rm{ [keV] } $', fontsize = size_font )
plt.savefig( './../plots/pseudo_calculations/Fermi--Ep_distribution--simulated_2.png' )
plt.clf()
plt.close()



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

print 'Those exclusive to Fermi	:	', indices_exclusive.size


Fermi_exclusive_name				=	Fermi_rest_name[indices_exclusive]
Fermi_exclusive_ID					=	Fermi_rest_ID[indices_exclusive]
Fermi_exclusive_Ttime				=	Fermi_rest_Tt[indices_exclusive]
Fermi_exclusive_T90					=	Fermi_rest_T90[indices_exclusive]
Fermi_exclusive_T90_error			=	Fermi_rest_T90_error[indices_exclusive]
Fermi_exclusive_flux				=	Fermi_rest_flux[indices_exclusive]
Fermi_exclusive_flux_error			=	Fermi_rest_flux_error[indices_exclusive]
Fermi_exclusive_fluence				=	Fermi_rest_fluence[indices_exclusive]
Fermi_exclusive_fluence_error		=	Fermi_rest_fluence_error[indices_exclusive]
Fermi_exclusive_Epeak_in_MeV		=	Fermi_exclusive_Epeak_in_MeV[indices_exclusive]
Fermi_exclusive_Epeak_in_MeV_error	=	Fermi_exclusive_Epeak_in_MeV_error[indices_exclusive]


#~ Fermi_exclusive_table	=	Table( [ Fermi_exclusive_ID, Fermi_exclusive_name, Fermi_exclusive_T90, Fermi_exclusive_T90_error, Fermi_exclusive_flux, Fermi_exclusive_flux_error, Fermi_exclusive_Epeak_in_MeV, Fermi_exclusive_Epeak_in_MeV_error ], 
									#~ names = [ 'ID', 'name', 'T90', 'T90_error', 'flux [erg.cm^{-2}.s^{-1}]', 'flux_error [erg.cm^{-2}.s^{-1}]', 'Epeak (siml) [MeV]', 'Epeak_error (siml) [MeV]' ] )
#~ ascii.write( Fermi_exclusive_table, './../tables/Fermi_GRBs--exclusive.txt', format = 'fixed_width', overwrite = True )


Fermi_exclusive_table	=	Table( [Fermi_exclusive_name, Fermi_exclusive_Ttime, Fermi_exclusive_T90, Fermi_exclusive_T90_error, Fermi_exclusive_flux, Fermi_exclusive_flux_error, Fermi_exclusive_fluence, Fermi_exclusive_fluence_error ], 
									names = [ 'name', 'T-time', 'T90', 'T90_error', 'flux [erg.cm^{-2}.s^{-1}]', 'flux_error [erg.cm^{-2}.s^{-1}]', 'fluence [erg.cm^{-2}]', 'fluence_error [erg.cm^{-2}]' ] )
ascii.write( Fermi_exclusive_table, './../tables/Fermi_GRBs--without_spectral_parameters.txt', format = 'fixed_width', overwrite = True )



####################################################################################################################################################

