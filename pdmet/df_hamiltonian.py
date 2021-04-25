#!/usr/bin/env python -u 
'''
pDMET: Density Matrix Embedding theory for Periodic Systems
Copyright (C) 2018 Hung Q. Pham. All Rights Reserved.
A few functions in pDMET are modifed from QC-DMET Copyright (C) 2015 Sebastian Wouters

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Email: Hung Q. Pham <pqh3.14@gmail.com>
'''

import numpy as np
import scipy.optimize as optimize


'''DFT XC library'''
GGA_X = {
'GGA_X_GAM'                    : 32 , # H. S. Yu, W. Zhang, P. Verma, X. He, and D. G. Truhlar, Phys. Chem. Chem. Phys. 17, 12146 (2015)
'GGA_X_HCTH_A'                 : 34 , # F. A. Hamprecht, A. J. Cohen, D. J. Tozer, and N. C. Handy, J. Chem. Phys. 109, 6264 (1998)
'GGA_X_EV93'                   : 35 , # E. Engel and S. H. Vosko, Phys. Rev. B 47, 13164 (1993)
'GGA_X_BCGP'                   : 38 , # K. Burke, A. Cancio, T. Gould, and S. Pittalis, ArXiv e-prints (2014), arXiv:1409.4834 [cond-mat.mtrl-sci]
'GGA_X_LAMBDA_OC2_N'           : 40 , # M. M. Odashima, K. Capelle, and S. B. Trickey, J. Chem. Theory Comput. 5, 798 (2009)
'GGA_X_B86_R'                  : 41 , # I. Hamada, Phys. Rev. B 89, 121103 (2014)
'GGA_X_LAMBDA_CH_N'            : 44 , # M. M. Odashima, K. Capelle, and S. B. Trickey, J. Chem. Theory Comput. 5, 798 (2009)
'GGA_X_LAMBDA_LO_N'            : 45 , # M. M. Odashima, K. Capelle, and S. B. Trickey, J. Chem. Theory Comput. 5, 798 (2009)
'GGA_X_HJS_B88_V2'             : 46 , # E. Weintraub, T. M. Henderson, and G. E. Scuseria, J. Chem. Theory Comput. 5, 754 (2009)
'GGA_X_Q2D'                    : 48 , # L. Chiodo, L. A. Constantin, E. Fabiano, and F. Della Sala, Phys. Rev. Lett. 108, 126402 (2012)
'GGA_X_PBE_MOL'                : 49 , # J. M. del Campo, J. L. G\'azquez, S. B. Trickey, and A. Vela, J. Chem. Phys. 136, 104108 (2012)
'GGA_X_AK13'                   : 56 , # R. Armiento and S. Kummel, Phys. Rev. Lett. 111, 036402 (2013)
'GGA_X_LV_RPW86'               : 58 , # K. Berland and P. Hyldgaard, Phys. Rev. B 89, 035412 (2014)
'GGA_X_PBE_TCA'                : 59 , # V. Tognetti, P. Cortona, and C. Adamo, Chem. Phys. Lett. 460, 536 (2008)
'GGA_X_PBEINT'                 : 60 , # E. Fabiano, L. A. Constantin, and F. Della Sala, Phys. Rev. B 82, 113104 (2010)
'GGA_X_VMT84_GE'               : 68 , # A. Vela, J. C. Pacheco-Kato, J. L. G\'azquez, J. M. del Campo, and S. B. Trickey, J. Chem. Phys. 136, 144115 (2012)
'GGA_X_VMT84_PBE'              : 69 , # A. Vela, J. C. Pacheco-Kato, J. L. G\'azquez, J. M. del Campo, and S. B. Trickey, J. Chem. Phys. 136, 144115 (2012)
'GGA_X_VMT_GE'                 : 70 , # A. Vela, V. Medel, and S. B. Trickey, J. Chem. Phys. 130, 244103 (2009)
'GGA_X_VMT_PBE'                : 71 , # A. Vela, V. Medel, and S. B. Trickey, J. Chem. Phys. 130, 244103 (2009)
'GGA_X_N12'                    : 82 , # R. Peverati and D. G. Truhlar, J. Chem. Theory Comput. 8, 2310 (2012)
'GGA_X_SSB_SW'                 : 90 , # M. Swart, M. Sol\'a, and F. M. Bickelhaupt, J. Comput. Methods Sci. Eng. 9, 69 (2009)
'GGA_X_SSB'                    : 91 , # M. Swart, M. Sol\'a, and F. M. Bickelhaupt, J. Chem. Phys. 131, 094103 (2009)
'GGA_X_SSB_D'                  : 92 , # M. Swart, M. Sol\'a, and F. M. Bickelhaupt, J. Chem. Phys. 131, 094103 (2009)
'GGA_X_BPCCAC'                 : 98 , # E. Br\'emond, D. Pilard, I. Ciofini, H. Chermette, C. Adamo, and P. Cortona, Theor. Chem. Acc. 131, 1184 (2012)
'GGA_X_PBE'                    : 101, # J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
'GGA_X_PBE_R'                  : 102, # Y. Zhang and W. Yang, Phys. Rev. Lett. 80, 890 (1998)
'GGA_X_B86'                    : 103, # A. D. Becke, J. Chem. Phys. 84, 4524 (1986)
'GGA_X_HERMAN'                 : 104, # F. Herman, J. P. V. Dyke, and I. B. Ortenburger, Phys. Rev. Lett. 22, 807 (1969)
'GGA_X_B86_MGC'                : 105, # A. D. Becke, J. Chem. Phys. 84, 4524 (1986)
'GGA_X_B88'                    : 106, # A. D. Becke, Phys. Rev. A 38, 3098 (1988)
'GGA_X_G96'                    : 107, # P. M. W. Gill, Mol. Phys. 89, 433 (1996)
'GGA_X_PW86'                   : 108, # J. P. Perdew and W. Yue, Phys. Rev. B 33, 8800 (1986)
'GGA_X_PW91'                   : 109, # J. P. Perdew, in Proceedings of the 75. WE-Heraeus-Seminar and 21st Annual International Symposium on Electronic Structure of Solids, edited by P. Ziesche and H. Eschrig (Akademie Verlag, Berlin, 1991) p. 11
'GGA_X_OPTX'                   : 110, # N. C. Handy and A. J. Cohen, Mol. Phys. 99, 403 (2001)
'GGA_X_DK87_R1'                : 111, # A. E. DePristo and J. D. Kress, J. Chem. Phys. 86, 1425 (1987)
'GGA_X_DK87_R2'                : 112, # A. E. DePristo and J. D. Kress, J. Chem. Phys. 86, 1425 (1987)
'GGA_X_LG93'                   : 113, # D. J. Lacks and R. G. Gordon, Phys. Rev. A 47, 4681 (1993)
'GGA_X_FT97_A'                 : 114, # M. Filatov and W. Thiel, Mol. Phys. 91, 847 (1997)
'GGA_X_FT97_B'                 : 115, # M. Filatov and W. Thiel, Mol. Phys. 91, 847 (1997)
'GGA_X_PBE_SOL'                : 116, # J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, L. A. Constantin, X. Zhou, and K. Burke, Phys. Rev. Lett. 100, 136406 (2008)
'GGA_X_RPBE'                   : 117, # B. Hammer, L. B. Hansen, and J. K. Norskov, Phys. Rev. B 59, 7413 (1999)
'GGA_X_WC'                     : 118, # Z. Wu and R. E. Cohen, Phys. Rev. B 73, 235116 (2006)
'GGA_X_MPW91'                  : 119, # C. Adamo and V. Barone, J. Chem. Phys. 108, 664 (1998)
'GGA_X_AM05'                   : 120, # R. Armiento and A. E. Mattsson, Phys. Rev. B 72, 085108 (2005)
'GGA_X_PBEA'                   : 121, # G. K. H. Madsen, Phys. Rev. B 75, 195108 (2007)
'GGA_X_MPBE'                   : 122, # C. Adamo and V. Barone, J. Chem. Phys. 116, 5933 (2002)
'GGA_X_XPBE'                   : 123, # X. Xu and W. A. Goddard, J. Chem. Phys. 121, 4068 (2004)
'GGA_X_2D_B86_MGC'             : 124, # S. Pittalis, E. Rasanen, J. G. Vilhena, and M. A. L. Marques, Phys. Rev. A 79, 012503 (2009)
'GGA_X_BAYESIAN'               : 125, # J. J. Mortensen, K. Kaasbjerg, S. L. Frederiksen, J. K. Norskov, J. P. Sethna, and K. W. Jacobsen, Phys. Rev. Lett. 95, 216401 (2005)
'GGA_X_PBE_JSJR'               : 126, # L. S. Pedroza, A. J. R. da Silva, and K. Capelle, Phys. Rev. B 79, 201106 (2009)
'GGA_X_2D_B88'                 : 127, # J. G. Vilhena and M. A. L. Marques, unpublished (2014)
'GGA_X_2D_B86'                 : 128, # J. G. Vilhena and M. A. L. Marques, unpublished (2014)
'GGA_X_2D_PBE'                 : 129, # J. G. Vilhena and M. A. L. Marques, unpublished (2014)
'GGA_X_OPTB88_VDW'             : 139, # J. Klime\v{s}, D. R. Bowler, and A. Michaelides, J. Phys.: Condens. Matter 22, 022201 (2010)
'GGA_X_PBEK1_VDW'              : 140, # J. Klime\v{s}, D. R. Bowler, and A. Michaelides, J. Phys.: Condens. Matter 22, 022201 (2010)
'GGA_X_OPTPBE_VDW'             : 141, # J. Klime\v{s}, D. R. Bowler, and A. Michaelides, J. Phys.: Condens. Matter 22, 022201 (2010)
'GGA_X_RGE2'                   : 142, # A. Ruzsinszky, G. I. Csonka, and G. E. Scuseria, J. Chem. Theory Comput. 5, 763 (2009)
'GGA_X_RPW86'                  : 144, # E. D. Murray, K. Lee, and D. C. Langreth, J. Chem. Theory Comput. 5, 2754 (2009)
'GGA_X_KT1'                    : 145, # T. W. Keal and D. J. Tozer, J. Chem. Phys. 119, 3015 (2003)
'GGA_X_MB88'                   : 149, # V. Tognetti and C. Adamo, J. Phys. Chem. A 113, 14415 (2009)
'GGA_X_SOGGA'                  : 150, # Y. Zhao and D. G. Truhlar, J. Chem. Phys. 128, 184109 (2008)
'GGA_X_SOGGA11'                : 151, # R. Peverati, Y. Zhao, and D. G. Truhlar, J. Phys. Chem. Lett. 2, 1991 (2011)
'GGA_X_C09X'                   : 158, # V. R. Cooper, Phys. Rev. B 81, 161104 (2010)
'GGA_X_OL2'                    : 183, # P. Fuentealba and O. Reyes, Chem. Phys. Lett. 232, 31 (1995)
'GGA_X_APBE'                   : 184, # L. A. Constantin, E. Fabiano, S. Laricchia, and F. Della Sala, Phys. Rev. Lett. 106, 186406 (2011)
'GGA_X_HTBS'                   : 191, # P. Haas, F. Tran, P. Blaha, and K. Schwarz, Phys. Rev. B 83, 205117 (2011)
'GGA_X_AIRY'                   : 192, # L. A. Constantin, A. Ruzsinszky, and J. P. Perdew, Phys. Rev. B 80, 035125 (2009)
'GGA_X_LAG'                    : 193, # L. Vitos, B. Johansson, J. Koll\'ar, and H. L. Skriver, Phys. Rev. B 62, 10046 (2000)
'GGA_X_PBEFE'                  : 265, # R. Sarmiento-P\'erez, S. Botti, and M. A. L. Marques, J. Chem. Theory Comput. 11, 3844 (2015)
'GGA_X_CAP'                    : 270, # J. Carmona-Esp\'indola, J. L. G\'azquez, A. Vela, and S. B. Trickey, J. Chem. Phys. 142, 054105 (2015), 10.1063/1.4906606
'GGA_X_EB88'                   : 271, # P. Elliott and K. Burke, Can. J. Chem. 87, 1485 (2009)
'GGA_X_BEEFVDW'                : 285, # J. Wellendorff, K. T. Lundgaard, A. M\o{gelh\o{j}, V. Petzold, D. D. Landis, J. K. N\o{rskov}, T. Bligaard, and K. W. Jacobsen, }Phys. Rev. B 85, 235149 (2012)
'GGA_X_PBETRANS'               : 291, # Eric Bremond, I. Ciofini, and C. Adamo, Mol. Phys. 114, 1059 (2016)
'GGA_X_CHACHIYO'               : 298, # T. {Chachiyo and H. {Chachiyo}, }ArXiv e-prints (2017), arXiv:1706.01343 [cond-mat.mtrl-sci]
'GGA_X_WPBEH'                  : 524, # J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003)
'GGA_X_HJS_PBE'                : 525, # T. M. Henderson, B. G. Janesko, and G. E. Scuseria, J. Chem. Phys. 128, 194105 (2008)
'GGA_X_HJS_PBE_SOL'            : 526, # T. M. Henderson, B. G. Janesko, and G. E. Scuseria, J. Chem. Phys. 128, 194105 (2008)
'GGA_X_HJS_B97X'               : 528, # T. M. Henderson, B. G. Janesko, and G. E. Scuseria, J. Chem. Phys. 128, 194105 (2008)
'GGA_X_ITYH'                   : 529, # H. Iikura, T. Tsuneda, T. Yanai, and K. Hirao, J. Chem. Phys. 115, 3540 (2001)
'GGA_X_SFAT'                   : 530, # A. Savin and H.-J. Flad, Int. J. Quantum Chem. 56, 327 (1995)
'GGA_X_SG4'                    : 533, # L. A. Constantin, A. Terentjevs, F. Della Sala, P. Cortona, and E. Fabiano, Phys. Rev. B 93, 045126 (2016)
'GGA_X_GG99'                   : 535, # A. T. Gilbert and P. M. Gill, Chem. Phys. Lett. 312, 511 (1999)
'GGA_X_PBEPOW'                 : 539, # Eric Bremond, J. Chem. Phys. 145, 244102 (2016)
'GGA_X_KGG99'                  : 544, # A. T. Gilbert and P. M. Gill, Chem. Phys. Lett. 312, 511 (1999)
'GGA_X_B88M'                   : 570, # E. Proynov, H. Chermette, and D. R. Salahub, J. Chem. Phys. 113, 10013 (2000)
}

GGA_C = {
'GGA_C_GAM'                    : 33 , # H. S. Yu, W. Zhang, P. Verma, X. He, and D. G. Truhlar, Phys. Chem. Chem. Phys. 17, 12146 (2015)
'GGA_C_BCGP'                   : 39 , # K. Burke, A. Cancio, T. Gould, and S. Pittalis, ArXiv e-prints (2014), arXiv:1409.4834 [cond-mat.mtrl-sci]
'GGA_C_Q2D'                    : 47 , # L. Chiodo, L. A. Constantin, E. Fabiano, and F. Della Sala, Phys. Rev. Lett. 108, 126402 (2012)
'GGA_C_ZPBEINT'                : 61 , # L. A. Constantin, E. Fabiano, and F. Della Sala, Phys. Rev. B 84, 233103 (2011)
'GGA_C_PBEINT'                 : 62 , # E. Fabiano, L. A. Constantin, and F. Della Sala, Phys. Rev. B 82, 113104 (2010)
'GGA_C_ZPBESOL'                : 63 , # L. A. Constantin, E. Fabiano, and F. Della Sala, Phys. Rev. B 84, 233103 (2011)
'GGA_C_N12_SX'                 : 79 , # R. Peverati and D. G. Truhlar, Phys. Chem. Chem. Phys. 14, 16187 (2012)
'GGA_C_N12'                    : 80 , # R. Peverati and D. G. Truhlar, J. Chem. Theory Comput. 8, 2310 (2012)
'GGA_C_REGTPSS'                : 83 , # J. P. Perdew, A. Ruzsinszky, G. I. Csonka, L. A. Constantin, and J. Sun, Phys. Rev. Lett. 103, 026403 (2009)
'GGA_C_OP_XALPHA'              : 84 , # T. Tsuneda, T. Suzumura, and K. Hirao, J. Chem. Phys. 110, 10664 (1999)
'GGA_C_OP_G96'                 : 85 , # T. Tsuneda, T. Suzumura, and K. Hirao, J. Chem. Phys. 110, 10664 (1999)
'GGA_C_OP_PBE'                 : 86 , # T. Tsuneda, T. Suzumura, and K. Hirao, J. Chem. Phys. 110, 10664 (1999)
'GGA_C_OP_B88'                 : 87 , # T. Tsuneda, T. Suzumura, and K. Hirao, J. Chem. Phys. 110, 10664 (1999)
'GGA_C_FT97'                   : 88 , # M. Filatov and W. Thiel, Int. J. Quantum Chem. 62, 603 (1997)
'GGA_C_HCTH_A'                 : 97 , # F. A. Hamprecht, A. J. Cohen, D. J. Tozer, and N. C. Handy, J. Chem. Phys. 109, 6264 (1998)
'GGA_C_REVTCA'                 : 99 , # V. Tognetti, P. Cortona, and C. Adamo, Chem. Phys. Lett. 460, 536 (2008)
'GGA_C_TCA'                    : 100, # V. Tognetti, P. Cortona, and C. Adamo, J. Chem. Phys. 128, 034101 (2008)
'GGA_C_PBE'                    : 130, # J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
'GGA_C_LYP'                    : 131, # C. Lee, W. Yang, and R. G. Parr, Phys. Rev. B 37, 785 (1988)
'GGA_C_P86'                    : 132, # J. P. Perdew, Phys. Rev. B 33, 8822 (1986)
'GGA_C_PBE_SOL'                : 133, # J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, L. A. Constantin, X. Zhou, and K. Burke, Phys. Rev. Lett. 100, 136406 (2008)
'GGA_C_PW91'                   : 134, # J. P. Perdew, in Proceedings of the 75. WE-Heraeus-Seminar and 21st Annual International Symposium on Electronic Structure of Solids, edited by P. Ziesche and H. Eschrig (Akademie Verlag, Berlin, 1991) p. 11
'GGA_C_AM05'                   : 135, # R. Armiento and A. E. Mattsson, Phys. Rev. B 72, 085108 (2005)
'GGA_C_XPBE'                   : 136, # X. Xu and W. A. Goddard, J. Chem. Phys. 121, 4068 (2004)
'GGA_C_LM'                     : 137, # D. C. Langreth and M. J. Mehl, Phys. Rev. Lett. 47, 446 (1981)
'GGA_C_PBE_JRGX'               : 138, # L. S. Pedroza, A. J. R. da Silva, and K. Capelle, Phys. Rev. B 79, 201106 (2009)
'GGA_C_RGE2'                   : 143, # A. Ruzsinszky, G. I. Csonka, and G. E. Scuseria, J. Chem. Theory Comput. 5, 763 (2009)
'GGA_C_WL'                     : 147, # L. C. Wilson and M. Levy, Phys. Rev. B 41, 12930 (1990)
'GGA_C_WI'                     : 148, # L. C. Wilson and S. Ivanov, Int. J. Quantum Chem. 69, 523 (1998)
'GGA_C_SOGGA11'                : 152, # R. Peverati, Y. Zhao, and D. G. Truhlar, J. Phys. Chem. Lett. 2, 1991 (2011)
'GGA_C_WI0'                    : 153, # L. C. Wilson and S. Ivanov, Int. J. Quantum Chem. 69, 523 (1998)
'GGA_C_SOGGA11_X'              : 159, # R. Peverati and D. G. Truhlar, J. Chem. Phys. 135, 191102 (2011)
'GGA_C_APBE'                   : 186, # L. A. Constantin, E. Fabiano, S. Laricchia, and F. Della Sala, Phys. Rev. Lett. 106, 186406 (2011)
'GGA_C_OPTC'                   : 200, # A. J. Cohen and N. C. Handy, Mol. Phys. 99, 607 (2001)
'GGA_C_PBELOC'                 : 246, # L. A. Constantin, E. Fabiano, and F. Della Sala, Phys. Rev. B 86, 035130 (2012)
'GGA_C_PBEFE'                  : 258, # R. Sarmiento-P\'erez, S. Botti, and M. A. L. Marques, J. Chem. Theory Comput. 11, 3844 (2015)
'GGA_C_OP_PW91'                : 262, # T. Tsuneda, T. Suzumura, and K. Hirao, J. Chem. Phys. 110, 10664 (1999)
'GGA_C_PBE_MOL'                : 272, # J. M. del Campo, J. L. G\'azquez, S. B. Trickey, and A. Vela, J. Chem. Phys. 136, 104108 (2012)
'GGA_C_BMK'                    : 280, # A. D. Boese and J. M. L. Martin, J. Chem. Phys. 121, 3405 (2004)
'GGA_C_TAU_HCTH'               : 281, # A. D. Boese and N. C. Handy, J. Chem. Phys. 116, 9559 (2002)
'GGA_C_HYB_TAU_HCTH'           : 283, # A. D. Boese and N. C. Handy, J. Chem. Phys. 116, 9559 (2002)
'GGA_C_SG4'                    : 534, # L. A. Constantin, A. Terentjevs, F. Della Sala, P. Cortona, and E. Fabiano, Phys. Rev. B 93, 045126 (2016)
'GGA_C_SCAN_E0'                : 553, # J. Sun, A. Ruzsinszky, and J. P. Perdew, Phys. Rev. Lett. 115, 036402 (2015)
'GGA_C_GAPC'                   : 555, # E. Fabiano, P. E. Trevisanutto, A. Terentjevs, and L. A. Constantin, J. Chem. Theory Comput. 10, 2016 (2014), pMID: 26580528
'GGA_C_GAPLOC'                 : 556, # E. Fabiano, P. E. Trevisanutto, A. Terentjevs, and L. A. Constantin, J. Chem. Theory Comput. 10, 2016 (2014), pMID: 26580528
'GGA_C_ZVPBEINT'               : 557, # L. A. Constantin, E. Fabiano, and F. D. Sala, J. Chem. Phys. 137, 194105 (2012)
'GGA_C_ZVPBESOL'               : 558, # L. A. Constantin, E. Fabiano, and F. D. Sala, J. Chem. Phys. 137, 194105 (2012)
'GGA_C_TM_LYP'                 : 559, # A. J. Thakkar and S. P. McCarthy, J. Chem. Phys. 131, 134109 (2009)
'GGA_C_TM_PBE'                 : 560, # A. J. Thakkar and S. P. McCarthy, J. Chem. Phys. 131, 134109 (2009)
'GGA_C_W94'                    : 561, # L. C. Wilson, Chemical Physics 181, 337 (1994)
'GGA_C_CS1'                    : 565, # N. C. Handy a
}


'''Smaller set of XC to test'''
GGA_X = {

'GGA_X_PBE'                    : 101, # J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
'GGA_X_B88'                    : 106, # A. D. Becke, Phys. Rev. A 38, 3098 (1988)
'GGA_X_PW91'                   : 109, # J. P. Perdew, in Proceedings of the 75. WE-Heraeus-Seminar and 21st Annual International Symposium on Electronic Structure of Solids, edited by P. Ziesche and H. Eschrig (Akademie Verlag, Berlin, 1991) p. 11
'GGA_X_WC'                     : 118, # Z. Wu and R. E. Cohen, Phys. Rev. B 73, 235116 (2006)
'GGA_X_AM05'                   : 120, # R. Armiento and A. E. Mattsson, Phys. Rev. B 72, 085108 (2005)
}

GGA_C = {
'GGA_C_PBE'                    : 130, # J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
'GGA_C_LYP'                    : 131, # C. Lee, W. Yang, and R. G. Parr, Phys. Rev. B 37, 785 (1988)
'GGA_C_P86'                    : 132, # J. P. Perdew, Phys. Rev. B 33, 8822 (1986)
'GGA_C_AM05'                   : 135, # R. Armiento and A. E. Mattsson, Phys. Rev. B 72, 085108 (2005)
'GGA_C_SCAN_E0'                : 553, # J. Sun, A. Ruzsinszky, and J. P. Perdew, Phys. Rev. Lett. 115, 036402 (2015)
}


def get_init_uvec(xc_type='PBE0', dft_HF=None):
    '''Initialize uvec corresponding to each type of the DF-like cost function'''
    if dft_HF is not None:
        HF_percent = dft_HF
    else:
        HF_percent = 1.0

    if xc_type == 'PBE0':
        uvec = [HF_percent]  + [0.] * 2
    elif xc_type == 'RSH-PBE0':
        uvec = [1.] + [1.]  + [0.] * 2 
    elif xc_type == 'B3LYP':
        uvec = [HF_percent]  + [0.] * 4 
    elif xc_type == 'CAMB3LYP':
        uvec = [HF_percent , .0000001, 1. , 0., 0., 0., 0.]  
    elif xc_type == 'manyGGA':
        uvec = [HF_percent]  + [0.] * (len(GGA_X) + len(GGA_C)) 

    return uvec

def get_bounds(xc_type='PBE0', constraint=1, dft_HF=None):
    '''Prodive the bounds to optimize the DF-like cost function'''
    
    if constraint == 1:
        if xc_type == 'PBE0':
            lower = [-1.0] + [.0] * 2
            upper = [ 1.0] + [1.0] * 2
            bounds = optimize.Bounds(lower, upper)
        elif xc_type == 'RSH-PBE0':
            lower = [ -1.] * 2 + [.0] * 2
            upper = [ 1.0] * 4
            bounds = optimize.Bounds(lower, upper)
        elif xc_type == 'B3LYP':
            lower = [-1.0] + [.0] * 4
            upper = [ 1.0] + [1.0] * 4
            bounds = optimize.Bounds(lower, upper)
        elif xc_type == 'CAMB3LYP':
            bounds = optimize.Bounds([0.0,.0000001,.0,.0,.0,.0,.0], [1.,1.0,1.,1.,1.,1.,1.])
        elif xc_type == 'manyGGA':
            lower = [-1.0] + [.0] * (len(GGA_X) + len(GGA_C))
            upper = [ 1.0] + [.1] * (len(GGA_X) + len(GGA_C))
            bounds = optimize.Bounds(lower, upper)
    elif constraint == 2:
        if xc_type == 'PBE0':
            lower = [-1.0] * 3
            upper = [ 1.0] * 3
            bounds = optimize.Bounds(lower, upper)
        elif xc_type == 'RSH-PBE0':
            lower = [-1] * 4
            upper = [ 1.0]* 4
        elif xc_type == 'B3LYP':
            lower = [-1.0] * 5
            upper = [ 1.0] * 5
            bounds = optimize.Bounds(lower, upper)
        elif xc_type == 'CAMB3LYP':
            bounds = optimize.Bounds([0.0,.0000001,.0,.0,.0,.0,.0], [1.,1.0,1.,1.,1.,1.,1.])
        elif xc_type == 'manyGGA':
            lower = [-1.0] + [-1.0] * (len(GGA_X) + len(GGA_C))
            upper = [ 1.0] + [ 1.0] * (len(GGA_X) + len(GGA_C))
            bounds = optimize.Bounds(lower, upper)
    elif constraint == 3:
        if xc_type == 'manyGGA':
            lower = [ 0.0] * (len(GGA_X) + len(GGA_C))
            upper = [ 1.0] * (len(GGA_X) + len(GGA_C))
            bounds = optimize.Bounds(lower, upper)
    return bounds
      
    
def get_OEH_kpts(local, umat, xc_type='PBE0', dft_HF=None):
    '''Construct the mf Hamiltonian'''
    
    if dft_HF is not None and xc_type is not 'RSH-PBE0':
        umat[0] = np.float64(dft_HF)
        
    if xc_type == 'PBE0':
        xc =  "{:.12f}*HF + {:.12f}*PBE, {:.12f}*PBE".format(*umat)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm_kpts, 0, local.kpts, None)
        veff = vxc + local.vj - umat[0] * 0.5 * local.vk
        OEH_kpts = local.h_core + veff
        if local._is_KROHF:
            OEH_kpts = 0.5 * (OEH_kpts[0] + OEH_kpts[1])
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo)
        
    elif xc_type == 'B3LYP':
        xc =  "{:.12f}*HF + {:.12f}*LDA + {:.12f}*B88, {:.12f}*LYP + {:.12f}*VWN".format(*umat)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm_kpts, 0, local.kpts, None)
        veff = vxc + local.vj - umat[0] * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        if local._is_KROHF:
            OEH_kpts = 0.5 * (OEH_kpts[0] + OEH_kpts[1])
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo) 

    elif xc_type == 'RSH-PBE0':
        # TODO: right now the range-seperation parameter is fixed, it is super expensive to update it every cycle
        # because the vk long-range needed to be generated 
        umat = [local.xc_omega] + np.asarray(umat).tolist()
        xc =  "{1:.12f}*SR_HF({0:.2f})+ {2:.12f}*LR_HF({0:.2f}) + {3:.12f}*PBE, {4:.12f}*PBE".format(*umat)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm_kpts, 0, local.kpts, None)
        veff = vxc + local.vj - 0.5 * (umat[1]*local.vksr + umat[2]*local.vklr)
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo)
        
    elif xc_type == 'CAMB3LYP':
        # TODO: need to debug
        xc =  '{:.12f}*SR_HF({:.12f}) + {:.12f}*LR_HF({:.12f}) + {:.12f}*ITYH + {:.12f}*B88, {:.12f}*VWN5 + {:.12f}*LYP'.format(umat)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm_kpts, 0, local.kpts, None)
        veff = vxc + local.vj - umat[0] * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo) 

    elif xc_type == 'manyGGA':
        x = '{:.12f}*HF + {:.12f}*' + '+ {:.12f}*'.join(GGA_X.keys())
        c = ', {:.12f}*' + '+ {:.12f}*'.join(GGA_C.keys())
        xc = (x + c).format(*umat)
        n, exc, vxc = local.kks._numint.nr_rks(local.cell, local.kks.grids, xc, local.dm_kpts, 0, local.kpts, None)
        veff = vxc + local.vj - umat[0] * 0.5 * local.vk  
        OEH_kpts = local.h_core + veff
        OEH_kpts = local.ao_2_loc(OEH_kpts, local.ao2lo) 
    
    return OEH_kpts
    
    
    
      