from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
from scipy.signal import savgol_filter as sgf
from matplotlib  import cm
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10', size = 12)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.


z_max		=	1e+1


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



####################################################################################################################################################





####################################################################################################################################################


n	=	1.0
#	fBC0, in per solar-mass (M-sun)
fBC_min	=	2.59*1e-09
fBC_max	=	14.9*1e-09

#~ n	=	1.5
#~ #	fBC0, in per solar-mass (M-sun)
#~ fBC_min	=	1.47*1e-09
#~ fBC_max	=	6.84*1e-09

#~ n	=	2.0
#~ #	fBC0, in per solar-mass (M-sun)
#~ fBC_min	=	0.89*1e-09
#~ fBC_max	=	3.91*1e-09


angle_min = 3 ; angle_max = 26	# in degrees
ro	=	3	# digit to which to round off


####################################################################################################################################################





####################################################################################################################################################


def correct_theta( rate, theta ):
	
	beaming_factor	=	1 - np.cos( theta*(P/180) )
	return rate / beaming_factor


####################################################################################################################################################





####################################################################################################################################################


k_table		=	ascii.read( './../tables/k_correction.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL			=	k_table['dL'].data

CSFR_table	=	ascii.read( './../tables/CSFR_delayed--n={0:.1f}.txt'.format(n), format = 'fixed_width' )		;	global Phi
z			=	CSFR_table['z'].data
Psi			=	CSFR_table['CSFR_delayed'].data

vol_table	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )							;	global volume_term
vol			=	vol_table['vol'].data

#~ LHVKI = sio.loadmat( './../data/LHVKI-detected-BNS-sources.mat' )
#~ LHVKI_dL	= LHVKI['DL'][0]
#~ LHVKI_i		= LHVKI['iota'][0]
#~ LHVKI_SNR	= LHVKI['snrLHVKI'][0]
#~ print LHVKI_dL.size

data = sio.loadmat( './../data/BNS-sources-trial-10.mat' )['data']
#~ print data
SNR_LH		=	np.sqrt( data[:,5]**2 + data[:,6]**2 )												# LH
SNR_LHV		=	np.sqrt( data[:,5]**2 + data[:,6]**2 + data[:,7]**2 )								# LHV
SNR_LHVKI	=	np.sqrt( data[:,5]**2 + data[:,6]**2 + data[:,7]**2 + data[:,8]**2 + data[:,9]**2 )	# LHVKI
DL			=	data[:,0]
iota		=	data[:,4]
#~ print DL.size, iota.size, SNR_LHV.size, SNR_LHVKI.size

inds_LH		=	np.where( SNR_LH    > 8.0 )[0]
inds_LHV	=	np.where( SNR_LHV   > 8.0 )[0]
inds_LHVKI	=	np.where( SNR_LHVKI > 8.0 )[0]
#~ print inds_LH.size, inds_LHV.size, inds_LHVKI.size
#~ print '\n\n'
LH_dL		=	DL[inds_LH]
LH_i		=	iota[inds_LH]
LH_SNR		=	SNR_LHV[inds_LH]
LHV_dL		=	DL[inds_LHV]
LHV_i		=	iota[inds_LHV]
LHV_SNR		=	SNR_LHV[inds_LHV]
LHVKI_dL	=	DL[inds_LHVKI]
LHVKI_i		=	iota[inds_LHVKI]
LHVKI_SNR	=	SNR_LHVKI[inds_LHVKI]


#~ fig	=	plt.figure( figsize = (20, 6) )
#~ ax1	=	fig.add_subplot(131)
#~ ax1.set_title(r'$\rm{LH}$')
#~ ax1.set_xlim( -1, +1 )
#~ ax1.set_ylim( 100, 750 )
#~ ax1.set_yticks( range(150, 800, 150) )
#~ ax1.grid()
#~ ax1.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ im1	=	ax1.scatter( np.cos(LH_i), LH_dL, c = LH_SNR, cmap = cm.viridis )
#~ plt.ylabel( r'$ \rm{ d_{L} \; [Mpc] } $', fontsize = size_font+2 )
#~ ax2	=	fig.add_subplot(132)
#~ ax2.set_title(r'$\rm{LHV}$')
#~ ax2.set_xlim( -1, +1 )
#~ ax2.set_ylim( 100, 750 )
#~ ax2.set_yticks( range(150, 800, 150) )
#~ ax2.grid()
#~ ax2.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ im2	=	ax2.scatter( np.cos(LHV_i), LHV_dL, c = LHV_SNR, cmap = cm.viridis )
#~ ax3	=	fig.add_subplot(133)
#~ ax3.set_title(r'$\rm{LHVKI}$')
#~ ax3.set_xlim( -1, +1 )
#~ ax3.set_ylim( 100, 750 )
#~ ax3.set_yticks( range(150, 800, 150) )
#~ ax3.grid()
#~ ax3.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ im3	=	ax3.scatter( np.cos(LHVKI_i), LHVKI_dL, c = LHVKI_SNR, cmap = cm.viridis )
#~ im4	=	fig.colorbar( im3, ax = [ax1, ax2, ax3] )
#~ plt.savefig( './../plots/GW--dL_vs_i--all.png', bbox_inches = 'tight' )
#~ plt.savefig( './../plots/GW--dL_vs_i--all.pdf', bbox_inches = 'tight' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ fig	=	plt.figure( figsize = (15, 6) )
#~ ax1	=	fig.add_subplot(121)
#~ ax1.set_title(r'$\rm{LHV}$')
#~ ax1.set_xlim( -1, +1 )
#~ ax1.set_ylim( 100, 750 )
#~ ax1.set_yticks( range(150, 800, 150) )
#~ ax1.grid()
#~ ax1.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ im1	=	ax1.scatter( np.cos(LHV_i), LHV_dL, c = LHV_SNR, cmap = cm.viridis )
#~ plt.ylabel( r'$ \rm{ d_{L} \; [Mpc] } $', fontsize = size_font+2 )
#~ ax2	=	fig.add_subplot(122)
#~ ax2.set_title(r'$\rm{LHVKI}$')
#~ ax2.set_xlim( -1, +1 )
#~ ax2.set_ylim( 100, 750 )
#~ ax2.set_yticks( range(150, 800, 150) )
#~ ax2.grid()
#~ ax2.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ im2	=	ax2.scatter( np.cos(LHVKI_i), LHVKI_dL, c = LHVKI_SNR, cmap = cm.viridis )
#~ im3	=	fig.colorbar( im2, ax = [ax1, ax2] )
#~ plt.savefig( './../plots/GW--dL_vs_i--future.png', bbox_inches = 'tight' )
#~ plt.savefig( './../plots/GW--dL_vs_i--future.pdf', bbox_inches = 'tight' )
#~ plt.clf()
#~ plt.close()



####################################################################################################################################################






####################################################################################################################################################



#~ print LH_dL.size, LH_i.size, LH_SNR.size
#~ print LH_SNR.min()
inds = np.where( LH_SNR < 8.4 )[0]
#~ print inds.size

LH__truncated_cosi	=	np.cos( LHV_i[inds] )
LH__truncated_dL	=	LHV_dL[inds]
LH__truncated_cosi, LH__truncated_dL = mf.sort( LH__truncated_cosi, LH__truncated_dL )

#~ plt.title( 'LH' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.ylim( 0, 750 )
#~ plt.scatter( LH__truncated_cosi, LH__truncated_dL )
#~ plt.show()

N = 8
LH__truncated_cosi	=	mf.stats_over_array( LH__truncated_cosi, N )[1]
LH__truncated_dL	=	mf.stats_over_array(  LH__truncated_dL , N )[1]
LH__truncated_dL	=	sgf( LH__truncated_dL, 11, 2 )
LH__trunc_size		=	LH__truncated_dL.size
#~ print LH__trunc_size
#~ print '\n\n'

#~ plt.title( 'LH' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.xlim( -1.0, +1.0 )
#~ plt.ylim( 100, 750 )
#~ plt.plot( LH__truncated_cosi, LH__truncated_dL )
#~ plt.show()





#~ print LHV_dL.size, LHV_i.size, LHV_SNR.size
#~ print LHV_SNR.min()
inds = np.where( LHV_SNR < 8.05 )[0]
#~ print inds.size

LHV__truncated_cosi	=	np.cos( LHV_i[inds] )
LHV__truncated_dL	=	LHV_dL[inds]
LHV__truncated_cosi, LHV__truncated_dL = mf.sort( LHV__truncated_cosi, LHV__truncated_dL )

#~ plt.title( 'LHV' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.ylim( 0, 750 )
#~ plt.scatter( LHV__truncated_cosi, LHV__truncated_dL )
#~ plt.show()

N = 12
LHV__truncated_cosi	=	mf.stats_over_array( LHV__truncated_cosi, N )[1]
LHV__truncated_dL	=	mf.stats_over_array(  LHV__truncated_dL , N )[1]
LHV__truncated_dL	=	sgf( LHV__truncated_dL, 9, 2 )
LHV__trunc_size		=	LHV__truncated_dL.size
#~ print LHV__trunc_size
#~ print '\n\n'

#~ plt.title( 'LHV' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.xlim( -1.0, +1.0 )
#~ plt.ylim( 100, 750 )
#~ plt.plot( LHV__truncated_cosi, LHV__truncated_dL )
#~ plt.show()





#~ print LHVKI_dL.size, LHVKI_i.size, LHVKI_SNR.size
#~ print LHVKI_SNR.min()
inds = np.where( LHVKI_SNR < 8.015 )[0]
#~ print inds.size

LHVKI__truncated_cosi	=	np.cos( LHVKI_i[inds] )
LHVKI__truncated_dL		=	LHVKI_dL[inds]
LHVKI__truncated_cosi, LHVKI__truncated_dL = mf.sort( LHVKI__truncated_cosi, LHVKI__truncated_dL )

#~ plt.title( 'LHVKI' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.ylim( 0, 750 )
#~ plt.scatter( LHVKI__truncated_cosi, LHVKI__truncated_dL )
#~ plt.show()

N = 8
LHVKI__truncated_cosi	=	mf.stats_over_array( LHVKI__truncated_cosi, N )[1]
LHVKI__truncated_dL		=	mf.stats_over_array(  LHVKI__truncated_dL , N )[1]
LHVKI__truncated_dL		=	sgf( LHVKI__truncated_dL, 7, 2 )
LHVKI__trunc_size		=	LHVKI__truncated_dL.size
#~ print LHVKI__trunc_size
#~ print '\n\n'

#~ plt.title( 'LHVKI' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ d_L \; [Mpc] } $', fontsize = size_font+2 )
#~ plt.xlim( -1.0, +1.0 )
#~ plt.ylim( 100, 750 )
#~ plt.plot( LHVKI__truncated_cosi, LHVKI__truncated_dL )
#~ plt.show()



####################################################################################################################################################




####################################################################################################################################################


R0_min	=	round( fBC_min * Psi[0] * 1e9, ro-1 )
R0_max	=	round( fBC_max * Psi[0] * 1e9, ro-1 )
print 'R0 [ yr^{-1} Gpc^{-3} ] :              min & max : ', R0_min, R0_max, '\n'

print 'Correcting for beaming, to get :       min & max : ', round( correct_theta(R0_min, angle_max), ro-1 ), round( correct_theta(R0_max, angle_min), ro-1 ), '\n\n' 


####################################################################################################################################################




####################################################################################################################################################

ind_zMax	=	mf.nearest( z   , z_max)
z			=	z[  :ind_zMax]
Psi			=	Psi[:ind_zMax]

ind_zMax	=	mf.nearest(z_sim, z_max)
z_sim		=	z_sim[: ind_zMax]
dL			=	dL[   : ind_zMax]
vol			=	vol[  : ind_zMax]

####################################################################################################################################################




####################################################################################################################################################



LH__integrated_rates_min__without_correction	=	np.zeros(LH__trunc_size)
LH__integrated_rates_max__without_correction	=	LH__integrated_rates_min__without_correction.copy()
LH__integrated_rates_min__corrected         	=	LH__integrated_rates_min__without_correction.copy()
LH__integrated_rates_max__corrected         	=	LH__integrated_rates_min__without_correction.copy()

for k, max_dist in enumerate( LH__truncated_dL ):
	
	ind	= mf.nearest( dL, max_dist )
	
	trunc_z			=	z[  : ind]
	trunc_dL		=	dL[ : ind]
	trunc_Psi		=	Psi[: ind]
	trunc_vol		=	vol[: ind]
	
	LH__integrated_rate_min__without_correction	=	round( simps( fBC_min*trunc_Psi*trunc_vol, trunc_z ), ro )
	LH__integrated_rate_max__without_correction	=	round( simps( fBC_max*trunc_Psi*trunc_vol, trunc_z ), ro )
	LH__integrated_rates_min__without_correction[k]	=	LH__integrated_rate_min__without_correction
	LH__integrated_rates_max__without_correction[k]	=	LH__integrated_rate_max__without_correction
	
	LH__integrated_rates_min__corrected[k]	=	round( correct_theta(LH__integrated_rate_min__without_correction, angle_max), ro )
	LH__integrated_rates_max__corrected[k]	=	round( correct_theta(LH__integrated_rate_max__without_correction, angle_min), ro )


#~ plt.title( 'LH' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ N_{merger} \; [yr^{-1}] } $', fontsize = size_font+2 )
#~ plt.semilogy( LH__truncated_cosi, LH__integrated_rates_min__corrected, label = r'$ \rm{min} $' )
#~ plt.semilogy( LH__truncated_cosi, LH__integrated_rates_max__corrected, label = r'$ \rm{max} $' )
#~ plt.legend()
#~ plt.show()





LHV__integrated_rates_min__without_correction	=	np.zeros(LHV__trunc_size)
LHV__integrated_rates_max__without_correction	=	LHV__integrated_rates_min__without_correction.copy()
LHV__integrated_rates_min__corrected         	=	LHV__integrated_rates_min__without_correction.copy()
LHV__integrated_rates_max__corrected         	=	LHV__integrated_rates_min__without_correction.copy()

for k, max_dist in enumerate( LHV__truncated_dL ):
	
	ind	= mf.nearest( dL, max_dist )
	
	trunc_z			=	z[  : ind]
	trunc_dL		=	dL[ : ind]
	trunc_Psi		=	Psi[: ind]
	trunc_vol		=	vol[: ind]
	
	LHV__integrated_rate_min__without_correction	=	round( simps( fBC_min*trunc_Psi*trunc_vol, trunc_z ), ro )
	LHV__integrated_rate_max__without_correction	=	round( simps( fBC_max*trunc_Psi*trunc_vol, trunc_z ), ro )
	LHV__integrated_rates_min__without_correction[k]	=	LHV__integrated_rate_min__without_correction
	LHV__integrated_rates_max__without_correction[k]	=	LHV__integrated_rate_max__without_correction
	
	LHV__integrated_rates_min__corrected[k]	=	round( correct_theta(LHV__integrated_rate_min__without_correction, angle_max), ro )
	LHV__integrated_rates_max__corrected[k]	=	round( correct_theta(LHV__integrated_rate_max__without_correction, angle_min), ro )


#~ plt.title( 'LHV' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ N_{merger} \; [yr^{-1}] } $', fontsize = size_font+2 )
#~ plt.semilogy( LHV__truncated_cosi, LHV__integrated_rates_min__corrected, label = r'$ \rm{min} $' )
#~ plt.semilogy( LHV__truncated_cosi, LHV__integrated_rates_max__corrected, label = r'$ \rm{max} $' )
#~ plt.legend()
#~ plt.show()





LHVKI__integrated_rates_min__without_correction	=	np.zeros(LHVKI__trunc_size)
LHVKI__integrated_rates_max__without_correction	=	LHVKI__integrated_rates_min__without_correction.copy()
LHVKI__integrated_rates_min__corrected         	=	LHVKI__integrated_rates_min__without_correction.copy()
LHVKI__integrated_rates_max__corrected         	=	LHVKI__integrated_rates_min__without_correction.copy()

for k, max_dist in enumerate( LHVKI__truncated_dL ):
	
	ind	= mf.nearest( dL, max_dist )
	
	trunc_z			=	z[  : ind]
	trunc_dL		=	dL[ : ind]
	trunc_Psi		=	Psi[: ind]
	trunc_vol		=	vol[: ind]
	
	LHVKI__integrated_rate_min__without_correction	=	round( simps( fBC_min*trunc_Psi*trunc_vol, trunc_z ), ro )
	LHVKI__integrated_rate_max__without_correction	=	round( simps( fBC_max*trunc_Psi*trunc_vol, trunc_z ), ro )
	LHVKI__integrated_rates_min__without_correction[k]	=	LHVKI__integrated_rate_min__without_correction
	LHVKI__integrated_rates_max__without_correction[k]	=	LHVKI__integrated_rate_max__without_correction
	
	LHVKI__integrated_rates_min__corrected[k]	=	round( correct_theta(LHVKI__integrated_rate_min__without_correction, angle_max), ro )
	LHVKI__integrated_rates_max__corrected[k]	=	round( correct_theta(LHVKI__integrated_rate_max__without_correction, angle_min), ro )


#~ plt.title( 'LHVKI' )
#~ plt.xlabel( r'$ \cos i $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ N_{merger} \; [yr^{-1}] } $', fontsize = size_font+2 )
#~ plt.semilogy( LHVKI__truncated_cosi, LHVKI__integrated_rates_min__corrected, label = r'$ \rm{min} $' )
#~ plt.semilogy( LHVKI__truncated_cosi, LHVKI__integrated_rates_max__corrected, label = r'$ \rm{max} $' )
#~ plt.legend()
#~ plt.show()




####################################################################################################################################################




####################################################################################################################################################


fig	=	plt.figure()
ax	=	fig.add_subplot(111)
ax.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ N_{merger} \; [yr^{-1}] } $', fontsize = size_font+2, labelpad = padding+2 )
ax.yaxis.set_ticks_position('both')
ax.tick_params( axis = 'y', which = 'both', labelright = 'on' )
ax.plot(   LHV__truncated_cosi,   LHV__integrated_rates_min__corrected, 'k-' , label = 'LHV'   )
ax.plot( LHVKI__truncated_cosi, LHVKI__integrated_rates_min__corrected, 'r--', label = 'LHVKI' )
leg	=	ax.legend( loc = 'upper center' )
leg.get_frame().set_edgecolor('k')
plt.savefig( './../plots/lower_limits--future.png' )
plt.savefig( './../plots/lower_limits--future.pdf' )
plt.clf()
plt.close()

fig	=	plt.figure()
ax	=	fig.add_subplot(111)
ax.set_xlabel( r'$ \cos i $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ N_{merger} \; [yr^{-1}] } $', fontsize = size_font+2, labelpad = padding+2 )
ax.set_ylim( 2e-1, 1e+1 )
ax.yaxis.set_ticks_position('both')
ax.tick_params( axis = 'y', which = 'both', labelright = 'on' )
ax.set_yscale('log')
ax.plot(    LH__truncated_cosi,    LH__integrated_rates_min__corrected, linestyle = '--', linewidth = 3, label = 'LH'    )
ax.plot(   LHV__truncated_cosi,   LHV__integrated_rates_min__corrected, linestyle = '-' , linewidth = 3, label = 'LHV'   )
ax.plot( LHVKI__truncated_cosi, LHVKI__integrated_rates_min__corrected, linestyle = ':' , linewidth = 3, label = 'LHVKI' )
leg	=	ax.legend( loc = 'upper center' )
leg.get_frame().set_edgecolor('k')
plt.savefig( './../plots/lower_limits--all.png' )
plt.savefig( './../plots/lower_limits--all.pdf' )
plt.clf()
plt.close()

print 'Hard lower limit at LH    : ', round(  simps(    LH__integrated_rates_min__corrected,    LH__truncated_cosi ),  ro-1  )
print 'Hard lower limit at LHV   : ', round(  simps(   LHV__integrated_rates_min__corrected,   LHV__truncated_cosi ),  ro-1  )
print 'Hard lower limit at LHVKI : ', round(  simps( LHVKI__integrated_rates_min__corrected, LHVKI__truncated_cosi ),  ro-1  )


####################################################################################################################################################
