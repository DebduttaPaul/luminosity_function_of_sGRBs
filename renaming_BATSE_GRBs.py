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
A___mybestfit	=	2.946		#	my best-fit
eta_mybestfit	=	1.718		#	my best-fit

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


####################################################################################################################################################






####################################################################################################################################################



BATSE_GRBs_table			=	ascii.read( './../data/BATSE_GRBs--all.txt', format = 'fixed_width' )
BATSE_name					=	BATSE_GRBs_table['name'].data
BATSE_Ttime					=	BATSE_GRBs_table['time'].data
BATSE_T90					=	BATSE_GRBs_table['t90'].data
BATSE_T90_error				=	BATSE_GRBs_table['t90_error'].data
BATSE_flux					=	BATSE_GRBs_table['flux_1024'].data
BATSE_flux_error			=	BATSE_GRBs_table['flux_1024_error'].data
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



for j, name in enumerate(BATSE_name):	BATSE_name[j] = name[3:9]

temp_array	=	BATSE_name.copy()
for j, GRB in enumerate(BATSE_name):
	if ( np.delete(BATSE_name, j) == GRB ).any() == True:
		ind	=	np.where( BATSE_name == GRB )[0]
		if ind.size == 2:
			temp_array[ ind[0] ]	=	GRB + 'B'
			temp_array[ ind[1] ]	=	GRB + 'A'
		if ind.size == 3:
			temp_array[ ind[0] ]	=	GRB + 'C'
			temp_array[ ind[1] ]	=	GRB + 'B'
			temp_array[ ind[2] ]	=	GRB + 'A'
		if ind.size == 4:
			temp_array[ ind[0] ]	=	GRB + 'D'
			temp_array[ ind[1] ]	=	GRB + 'C'
			temp_array[ ind[2] ]	=	GRB + 'B'
			temp_array[ ind[3] ]	=	GRB + 'A'
		if ind.size == 5:
			temp_array[ ind[0] ]	=	GRB + 'E'
			temp_array[ ind[1] ]	=	GRB + 'D'
			temp_array[ ind[2] ]	=	GRB + 'C'
			temp_array[ ind[3] ]	=	GRB + 'B'
			temp_array[ ind[4] ]	=	GRB + 'A'
		if ind.size == 6:
			temp_array[ ind[0] ]	=	GRB + 'F'
			temp_array[ ind[1] ]	=	GRB + 'E'
			temp_array[ ind[2] ]	=	GRB + 'D'
			temp_array[ ind[3] ]	=	GRB + 'C'
			temp_array[ ind[4] ]	=	GRB + 'B'
			temp_array[ ind[5] ]	=	GRB + 'A'
		if ind.size == 7:
			temp_array[ ind[0] ]	=	GRB + 'G'
			temp_array[ ind[1] ]	=	GRB + 'F'
			temp_array[ ind[2] ]	=	GRB + 'E'
			temp_array[ ind[3] ]	=	GRB + 'D'
			temp_array[ ind[4] ]	=	GRB + 'C'
			temp_array[ ind[5] ]	=	GRB + 'B'
			temp_array[ ind[6] ]	=	GRB + 'A'
		if ind.size == 8:
			temp_array[ ind[0] ]	=	GRB + 'H'
			temp_array[ ind[1] ]	=	GRB + 'G'
			temp_array[ ind[2] ]	=	GRB + 'F'
			temp_array[ ind[3] ]	=	GRB + 'E'
			temp_array[ ind[4] ]	=	GRB + 'D'
			temp_array[ ind[5] ]	=	GRB + 'C'
			temp_array[ ind[6] ]	=	GRB + 'B'
			temp_array[ ind[7] ]	=	GRB + 'A'
		if ind.size == 9:
			temp_array[ ind[0] ]	=	GRB + 'I'
			temp_array[ ind[1] ]	=	GRB + 'H'
			temp_array[ ind[2] ]	=	GRB + 'G'
			temp_array[ ind[3] ]	=	GRB + 'F'
			temp_array[ ind[4] ]	=	GRB + 'E'
			temp_array[ ind[5] ]	=	GRB + 'D'
			temp_array[ ind[6] ]	=	GRB + 'C'
			temp_array[ ind[7] ]	=	GRB + 'B'
			temp_array[ ind[8] ]	=	GRB + 'A'
BATSE_name	=	temp_array.copy()

test_table	=	Table( [BATSE_name, BATSE_T90], names = ['name', 'T90'] )
ascii.write( test_table, './../tables/test_BATSE_names.txt', format = 'fixed_width', overwrite = True )


inds						=	np.where( np.ma.getmask( BATSE_T90 ) == False )
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
print 'With T90-measurements	:	' , BATSE_num
inds						=	np.where( np.ma.getmask( BATSE_flux ) == False )
BATSE_name					=	BATSE_name[inds]
BATSE_Ttime					=	BATSE_Ttime[inds]
BATSE_T90					=	BATSE_T90[inds]				#	in sec
BATSE_T90_error				=	BATSE_T90_error[inds]		#	in sec
BATSE_flux					=	BATSE_flux[inds]			#	in ph.cm^{-2}.s^{-1}
BATSE_flux_error			=	BATSE_flux_error[inds]		#	in ph.cm^{-2}.s^{-1}
BATSE_fluence1				=	BATSE_fluence1[inds]		#	in ph.cm^{-2}
BATSE_fluence1_error		=	BATSE_fluence1_error[inds]	#	in ph.cm^{-2}
BATSE_fluence2				=	BATSE_fluence2[inds]		#	in ph.cm^{-2}
BATSE_fluence2_error		=	BATSE_fluence2_error[inds]	#	in ph.cm^{-2}
BATSE_fluence3				=	BATSE_fluence3[inds]		#	in ph.cm^{-2}
BATSE_fluence3_error		=	BATSE_fluence3_error[inds]	#	in ph.cm^{-2}
BATSE_fluence4				=	BATSE_fluence4[inds]		#	in ph.cm^{-2}
BATSE_fluence4_error		=	BATSE_fluence4_error[inds]	#	in ph.cm^{-2}
BATSE_num					=	BATSE_name.size
print 'With flux-measurements	:	' , BATSE_num


BATSE_GRBs_table	=	Table( [ BATSE_name, BATSE_Ttime, BATSE_T90, BATSE_T90_error, BATSE_flux, BATSE_flux_error, BATSE_fluence1, BATSE_fluence1_error, BATSE_fluence2, BATSE_fluence2_error, BATSE_fluence3, BATSE_fluence3_error, BATSE_fluence4, BATSE_fluence4_error ], names = [ 'name', 'T-time', 'T90', 'T90_error', 'flux', 'flux_error', 'fluence_1', 'fluence_1_error', 'fluence_2', 'fluence_2_error', 'fluence_3', 'fluence_3_error', 'fluence_4', 'fluence_4_error' ] )
ascii.write( BATSE_GRBs_table, './../tables/BATSE_GRBs--measured.txt', format = 'fixed_width', overwrite = True )
