# BlaSim

A software to simulate RR Lyrae light curves undergoing the Blazhko effect.

Building on RR Lyrae (RRab, RRc) templates from Sesar et al. (2010), the Blazhko effect is simulated by using a combination of amplitude and
frequency modulation.

To simulate light curves from the LSST survey, the light curves are calculated on the appropriate cadences.

Photometric uncertainties are calculated according to the photometric error model of LSST (Ivezic et al. 2019)

$\sigma^2_{LSST} = \sigma^2_{sys}+\sigma^2_{rand}$

where $\sigma_{sys} = 0.005$ is the systematic error due to imperfect modeling of a point source, and 

$\sigma_{rand} = (1/25 - \gamma)*X + \gamma *X^2$

is the photometric error where $X = 10^{0.4*(m-m_5)}$,

with $m_5$ and
$\gamma$ being band-specific parameters (see Ivezic et al. 2019 Table 2).

Finally, the observed light curve is obtained as

$y_i = l_i + G(0,\sigma_{LSST}(l_i))$

where l is the simulated light curve without uncertainties.

A similar approach is used in Kovačević et al. (2021) for AGN light curves.		



Parameters to be modified by the user:
-------------------------------------

plot_flag			flag indicates if plots should be generated

cadence_u_filename,		path to cadence input file
cadence_g_filename,
cadence_r_filename,
cadence_i_filename,
cadence_z_filename
	
rHJD0 				phase offset; if set to -1: randomize; good parameters: 0 to 1

period_blazhko		Blazhko effect amplitude modulation; ifset to -1: randomize; good parameters: 10 - 100

fac_fm				period for the Blazhko effect frequency modulation; if set to -1: randomize; good parameters: 3.0 to 7.0
b				modulation index describes the factor by which the period is stretched at maximum; if set to -1: randomize; good parameters: 1.0 to 6.0


references:
----------------------------

Sesar, B., Ivezić, Z., Grammer, S. H., et al. 2010, ApJ, 708, 7
https://ui.adsabs.harvard.edu/abs/2010ApJ...708..717S/abstract

Ivezic, Z., Kahn, S. M., Tyson, A., Abel, b., Acosta, E., et al. 2019, ApJ, 873, 44
https://iopscience.iop.org/article/10.3847/1538-4357/ab042c

Kovačević, A., Ilić, D., Popović, L. C., et al. 2021
https://arxiv.org/pdf/2105.14889.pdf
