from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
from scipy import interpolate
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


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.


####################################################################################################################################################




####################################################################################################################################################


CSFR_delayed_dict	=	{}
for i in np.arange(1.0, 3.1, 0.5):
	CSFR_table				=	ascii.read( './../tables/CSFR_delayed--n={0:.1f}.txt'.format(i), format = 'fixed_width' )
	z						=	CSFR_table['z'].data
	CSFR					=	CSFR_table['CSFR'].data
	CSFR_delayed_dict[i]	=	CSFR_table['CSFR_delayed'].data


####################################################################################################################################################




####################################################################################################################################################



z_min = 0.0 ; z_max = 11.5 ; z_bin = 0.5
y_min = -3.5 ; y_max = 1.0 ; y_bin = 0.2
fig	=	plt.figure()
ax	=	fig.add_subplot(111)
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_min, y_max )
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ M_{\odot} yr^{-1} Mpc^{-3} } $', fontsize = size_font, labelpad = padding-6 )
major_ticks = np.arange( z_min, z_max,     2 )
minor_ticks = np.arange( z_min, z_max, z_bin )
ax.set_xticks( major_ticks )                                  
ax.set_xticks( minor_ticks, minor = True )
major_ticks = np.arange( y_min, y_max,     1 )
minor_ticks = np.arange( y_min, y_max, y_bin )
ax.set_yticks( major_ticks )
ax.set_yticks( minor_ticks, minor = True )
ax.plot( z, np.log10(CSFR), 'k-' , label = r'$ \rm{ log } \; \dot{ \rho_{\star} } $' )
for i in np.arange(1, 3.1, 0.5):	ax.plot( z, np.log10(CSFR_delayed_dict[i]), linestyle = '--', label = r'$ \rm{ log } \; \Psi , \, n = \, $' + r'$ {0:.1f} $'.format(i) )
plt.legend()
plt.savefig( './../plots/CSFR_delay.png' )
plt.savefig( './../plots/CSFR_delay.pdf' )
plt.clf()
plt.close()
#~ plt.show()
