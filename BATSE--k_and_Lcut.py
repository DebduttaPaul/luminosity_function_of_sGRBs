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

alpha_fix	=	-0.566
beta_fix	=	-2.823
Ec_fix		=	181.338	#	in keV.


BAT_sensitivity		=	0.20		# in  ph.s^{-1}.cm^{2}.
BATSE_sensitivity	=	0.10		# in  ph.s^{-1}.cm^{2}.


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.


####################################################################################################################################################




####################################################################################################################################################


k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
k_BATSE		=	k_table['k_BATSE'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
k_CZTI		=	k_table['k_CZTI' ].data
term_BATSE	=	k_table['term_BATSE'].data
term_Fermi	=	k_table['term_Fermi'].data
term_Swift	=	k_table['term_Swift'].data

threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__BATSE	=	threshold_data['L_cut__BATSE'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data
L_cut__CZTI		=	threshold_data['L_cut__CZTI' ].data


####################################################################################################################################################




####################################################################################################################################################


t0	=	time.time()

def integral_3rdterm__BATSE( E ):
	return E * sf.S( E, alpha_fix, beta_fix, Ec_fix )
def integrand_3rdterm__BATSE( E ):
	return sf.S( E, alpha_fix, beta_fix, Ec_fix )


E_min		=	20		#	in keV, BATSE band lower energy.
E_max		=	2e3		#	in keV, BATSE band upper energy.
d_chi		=	np.zeros( z_sim.size )
term_BATSE	=	d_chi.copy()
for j, z in enumerate( z_sim ):
	d_chi[j]		=	sf.chi(z)
	term_BATSE[j]	=	quad( integrand_3rdterm__BATSE, (1+z)*E_min, (1+z)*E_max )[0]
dL_sim		=	d_chi * ( 1 + z_sim )
term_BATSE	=	( integral_3rdterm__BATSE(E_min) + integral_3rdterm__BATSE(E_max) ) / term_BATSE
print 'BATSE distance and error terms done in mins:	' , ( time.time() - t0 ) / 60, '\n'


k_BATSE_sim	=	z_sim.copy()
for j, z in enumerate(z_sim):
	k_BATSE_sim[j]	=	sf.k_correction_factor_with_fixed_spectral_parameters__BATSE(z)
print 'BATSE k done in mins:				' , ( time.time() - t0 ) / 60

k_table	=	Table( [z_sim, dL_sim, k_BATSE_sim, k_Fermi, k_Swift, k_CZTI, term_BATSE, term_Fermi, term_Swift], names = ['z', 'dL', 'k_BATSE', 'k_Fermi', 'k_Swift', 'k_CZTI', 'term_BATSE', 'term_Fermi', 'term_Swift'] )
ascii.write( k_table, './../tables/k_table.txt', format = 'fixed_width', overwrite = True )


L_cut__BATSE=	z_sim.copy()
for j, z in enumerate(z_sim):
	L_cut__BATSE[j]	=	sf.Liso_with_fixed_spectral_parameters__BATSE( BATSE_sensitivity, z )
L_cut__BATSE	=	L_cut__BATSE * (cm_per_Mpc**2) * erg_per_keV
print '\n\n', 'Done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 )

threshold_data	=	Table( [z_sim, L_cut__BATSE, L_cut__Fermi, L_cut__Swift, L_cut__CZTI], names = ['z_sim', 'L_cut__BATSE', 'L_cut__Fermi', 'L_cut__Swift', 'L_cut__CZTI'] )
ascii.write( threshold_data, './../tables/thresholds.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################




####################################################################################################################################################


ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ k \, [ } 210 \rm{ \, keV] } $', fontsize = size_font, labelpad = padding-6 )
ax.plot( z_sim, k_BATSE/210, 'y-' , ms = marker_size, label = r'$ BATSE $' )
ax.plot( z_sim, k_Swift/210, 'b--', ms = marker_size, label = r'$ Swift $' )
ax.plot( z_sim, k_CZTI /210, 'g--', ms = marker_size, label = r'$ CZTI  $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/k_correction--BATSE,_Swift_and_CZTI.png' )
plt.savefig( './../plots/k_correction--BATSE,_Swift_and_CZTI.pdf' )
plt.clf()
plt.close()


ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ L_{cut} \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
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
