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

logL_bin	=	0.5
logL_min	=	-5
logL_max	=	+5


GBM_sensitivity		=	1e-8 * 8.0	# in erg.s^{-1}.cm^{2}.
BAT_sensitivity		=	0.20		# in  ph.s^{-1}.cm^{2}.
CZT_sensitivity		=	0.20		# in  ph.s^{-1}.cm^{2}.
BATSE_sensitivity	=	0.10		# in  ph.s^{-1}.cm^{2}.


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


k_table		=	ascii.read( './../tables/k_correction.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data

L_cut__Fermi	=	z_sim.copy()
L_cut__Swift	=	z_sim.copy()
L_cut__BATSE	=	z_sim.copy()
L_cut__CZTI		=	z_sim.copy()

t0	=	time.time()
#~ 
#~ 
#~ print '...Fermi...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__Fermi[j]	=	sf.Liso_with_fixed_spectral_parameters__Fermi( GBM_sensitivity, z )
#~ L_cut__Fermi	=	L_cut__Fermi * (cm_per_Mpc**2)
#~ print 'done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
#~ 
#~ print '...Swift...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__Swift[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( BAT_sensitivity, z )
#~ L_cut__Swift	=	L_cut__Swift * (cm_per_Mpc**2) * erg_per_keV
#~ 
#~ print 'done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
print '...BATSE...'
for j, z in enumerate(z_sim):
	L_cut__BATSE[j]	=	sf.Liso_with_fixed_spectral_parameters__BATSE( BATSE_sensitivity, z )
L_cut__BATSE	=	L_cut__BATSE * (cm_per_Mpc**2) * erg_per_keV
print 'done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
#~ 
#~ print '...CZTI...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__CZTI[j]	=	sf.Liso_with_fixed_spectral_parameters__CZTI( CZT_sensitivity, z )
#~ L_cut__CZTI		=	L_cut__CZTI  * (cm_per_Mpc**2) * erg_per_keV
#~ print 'done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
#~ 
#~ 
#~ threshold_data	=	Table( [z_sim, L_cut__BATSE, L_cut__Fermi, L_cut__Swift, L_cut__CZTI], names = ['z_sim', 'L_cut__BATSE', 'L_cut__Fermi', 'L_cut__Swift', 'L_cut__CZTI'] )
#~ ascii.write( threshold_data, './../tables/thresholds.txt', format = 'fixed_width', overwrite = True )
#~ 
#~ 
#~ ####################################################################################################################################################






####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__BATSE	=	threshold_data['L_cut__BATSE'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data
L_cut__CZTI		=	threshold_data['L_cut__CZTI' ].data


ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ L_{cut} \; $' + r'$ \rm{ [erg . s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.plot( z_sim, L_cut__BATSE, 'y-' , label = r'$ \rm{BATSE} $' )
ax.plot( z_sim, L_cut__Fermi, 'r-' , label = r'$    Fermi   $' )
ax.plot( z_sim, L_cut__Swift, 'b--', label = r'$    Swift   $' )
ax.plot( z_sim, L_cut__CZTI , 'g--', label = r'$  \rm{CZTI} $' )
plt.legend( loc = 'upper left' )
plt.savefig('./../plots/sensitivity_plot.png')
plt.savefig('./../plots/sensitivity_plot.pdf')
plt.clf()
plt.close()



####################################################################################################################################################
