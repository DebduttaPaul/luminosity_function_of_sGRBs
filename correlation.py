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
plt.rc('font', family = 'serif', serif = 'cm10', size = 12)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9


x_in_keV_min	=	2e01	;	x_in_keV_max	=	2e04
y_in_eps_min	=	1e48	;	y_in_eps_max	=	1e56

padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



####################################################################################################################################################




####################################################################################################################################################

def straight_line( x, m, c ):
	return m*x + c

####################################################################################################################################################




####################################################################################################################################################


table		=	ascii.read( './../data/combined_catalogue--literature.txt', format = 'fixed_width' )
GRB			=	table['GRB'].data
redshift	=	table['z'].data
Ep0			=	table['Ep0'].data
Ep0_poserr	=	table['Ep0_poserr'].data
Ep0_negerr	=	table['Ep0_negerr'].data
Lp			=	table['Lp'].data		* 1e-2
Lp_poserr	=	table['Lp_poserr'].data	* 1e-2
Lp_negerr	=	table['Lp_negerr'].data	* 1e-2

x_to_fit_in_keV			=	Ep0
x_to_fit_in_keV_poserr	=	Ep0_poserr
x_to_fit_in_keV_negerr	=	Ep0_negerr
x_to_fit_in_MeV			=	1e-3 * Ep0
x_to_fit_in_MeV_poserr	=	1e-3 * Ep0_poserr
x_to_fit_in_MeV_negerr	=	1e-3 * Ep0_negerr
y_to_fit				=	Lp
y_to_fit_poserr			=	Lp_poserr
y_to_fit_negerr			=	Lp_negerr


####################################################################################################################################################




####################################################################################################################################################

##	To check for correlations between the source quantities L_p and E_peak.
print 'Correlations...', '\n'
r, p_r	=	R( x_to_fit_in_MeV, y_to_fit )
s, p_s	=	S( x_to_fit_in_MeV, y_to_fit )
t, p_t	=	T( x_to_fit_in_MeV, y_to_fit )
print r, p_r
print s, p_s
print t, p_t
print '\n\n'


##	To extract the present best-fit.
x_to_fit_log	=	np.log10(x_to_fit_in_MeV)
y_to_fit_log	=	np.log10(y_to_fit)
popt, pcov = curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
eta_mybestfit	=	popt[0]	;	A___mybestfit	=	10**popt[1]
print '\n\n'
print 'Best-fit...', '\n'
print 'A   mean, error:	', round( A___mybestfit, 3 ), round( (10**pcov[1,1]-1)*A___mybestfit, 3 )
print 'eta mean, error:	', round( eta_mybestfit, 3 ), round( pcov[0,0], 3 )
print '\n\n'


ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ E_p \; \rm{ [keV] } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font )
ax.set_xlim( x_in_keV_min, x_in_keV_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( x_to_fit_in_keV, L_norm*y_to_fit, xerr = [x_to_fit_in_keV_poserr, x_to_fit_in_keV_negerr], yerr = [L_norm*y_to_fit_poserr, L_norm*y_to_fit_negerr], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
ax.plot( x_to_fit_in_keV, ( L_norm * A___mybestfit ) * ( x_to_fit_in_MeV**eta_mybestfit ), linestyle = '-', color = 'k' )
plt.savefig( './../plots/L_vs_Ep(1+z)--correlations.png' )
plt.savefig( './../plots/L_vs_Ep(1+z)--correlations.pdf' )
plt.clf()
plt.close()


####################################################################################################################################################

