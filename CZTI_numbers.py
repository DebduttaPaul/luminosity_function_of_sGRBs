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
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']




####################################################################################################################################################


GBM__total	=	298
BAT__total	=	122

tab	=	ascii.read( './../data/2yrs_CZTI_GRB_list.txt' )
SN	=	tab['SN'].data
GRB	=	tab['GRBname'].data
GBM	=	tab['GBM'].data
BAT	=	tab['BAT'].data
CZT	=	tab['CZT'].data
T90	=	(tab['T90'].data).astype(float)

GBM_only	=	np.where( (GBM=='yes') & (BAT=='no' ) )[0]	;	GBM_only__num	=	GBM_only.size
BAT_only	=	np.where( (BAT=='yes') & (GBM=='no' ) )[0]	;	BAT_only__num	=	BAT_only.size
both		=	np.where( (GBM=='yes') & (BAT=='yes') )[0]	;	both__num		=	both.size		;	GBM__num	=	GBM_only__num + both__num	;	BAT__num	=	BAT_only__num + both__num
none		=	np.where( (GBM=='no' ) & (BAT=='no' ) )[0]	;	none__num		=	none.size
print SN.size
#~ print GBM_only__num
#~ print BAT_only__num
#~ print both__num
#~ print none__num
print GBM__num
print BAT__num
print '\n'
#~ print GBM__num   / BAT__num
#~ print GBM__total / GBM__num
#~ print BAT__total / BAT__num
#~ print '\n'

ind	=	np.where(CZT=='yes')[0]
T90	=	T90[ind]
print ind.size
ind	=	np.where(T90<2.0)[0]
print ind.size




####################################################################################################################################################
