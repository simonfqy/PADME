from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os
import sys
import pwd
import pdb
import csv
import re
import math
import matplotlib.pyplot as plt
import time
import dcCustom
from dcCustom.feat import Protein

invalid_mols = [  
  "O=C(N([C@@H]([C@@H](OC)CC(=O)N1[C@@H]([C@@H](OC)[C@@H](C(=O)N[C@H](Cc2ccccc2)c2sccn2)C)CCC1)[C@@H](CC)C)C)[C@H](NC(=O)[C@H]([N+H](C)C)C(C)C)C(C)C",
  "O=C(Oc1c(C)c2OCOc2c2c1[C@@H]1SC[C@]3(C(=O)OCC2N2[C@@H](O)C4[N+H](C)C(c5c(O)c(OC)c(C)cc5C4)C12)[N+H2]CCc1c3cc(OC)c(O)c1)C",
  "O(C)c1c(OC)cc2c(c3c(c4c2C[C@@H]2[N+H](C4)CCCC2)ccc(OC)c3)c1",
  "O=C(N(C(C(=O)N1C(C(=O)N2C(C(=O)OC(C(=O)N3C(=O)C=C(OC)C3Cc3ccccc3)C(C)C)CCC2)CCC1)C(C)C)C)C(NC(=O)C([N+H](C)C)C(C)C)C(C)C",
  'O(C)C1(C)C(OC)C(OC2c3c([O-])c4C(=O)c5c([O-])cc6C7(C)C(O)C([N+H](C)C)C(O)C(O7)Oc6c5C(=O)c4cc3CC(O)(C)C2)OC(C)C1OC',
  "O=C(OC1C(C)C(O)C(C)CCC(OC)/C(/C)=C\CC(C)CC(OC)/C=C/CC(C(C(O)C(CCC(OC(=O)C([N+H](C)C)C)C(C(OC(=O)C)C(/C=C/N(C=O)C)C)C)C)C)OC(=O)/C=C/C=C/C1)C([N+H](C)C)COC",
  "O=C(OCC)Nc1[n+H]c(N)c2N=C([C@H](C)Nc2c1)c1ccccc1", 
  "FCCOC(=O)Nc1[n+H]c(N)c2N=C([C@H](C)Nc2c1)c1ccccc1",
  "O(C)[C@H]1[C@@H](C[C@H](O)C[N+H3])O[C@@H]2[C@@H]1CC(=O)C[C@@H]1O[C@@H]3[C@@H]4O[C@@]5(O[C@@H]6[C@H](O[C@H]3CC1)[C@H]4O[C@@H]6C5)CC[C@@H]1O[C@H](C(=C)C1)CC[C@H]1O[C@@H](C(=C)[C@H](C)C1)C2",
  "O=C1c2c([O-])c(C3OC(C)C(O)C([N+H](C)C)(C)C3)cc(C3OC(C)C(O)C([N+H](C)C)C3)c2C(=O)c2c1c1OC(C3(C)C(C4C(C)O4)O3)=CC(=O)c1c(C)c2",
  "O=C(N(C(C(=O)N1C(C(=O)N2C(C(=O)NCc3ccccc3)CCC2)CCC1)C(C)C)C)C(NC(=O)C([N+H](C)C)C(C)C)C(C)C",
  "O=[N+]([O-])c1c2c(NCCCCCNC(=O)CCC(NC(=O)C(NC(=O)COC3C(O)C(CO)OC(OCc4ccccc4)C3NC(=O)C)C)C(=O)N)c3c([n+H]c2ccc1)cccc3",
  "O=C(Oc1c(OC)ccc(/C=C\c2cc(OC)c(OC)c(OC)c2)c1)NCC[N+H](C)C",
  "O=C(OC(OC(=O)C)CCC[N+H2]C1C(O)C(C)OC(O[C@H]2c3c([O-])c4C(=O)c5c(OC)cccc5C(=O)c4c([O-])c3C[C@@](O)(C(=O)CO)C2)C1)C",
  "O=C(O[C@@H]1[C@H](Cc2ccc(OC)cc2)[N+H2]C[C@H]1O)C",
  "O=C(OC)[C@H]1[C@@](O)(CC)C[C@H](OC2OC(C)C(OC3OC(C)C(OC4OC(C)C(=O)CC4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])ccc([O-])c4C(=O)c3cc12",
  "O=C(OCC)Nc1[n+H]c(N)c2N=C(c3ccc(OC)cc3)CNc2c1",
  "O(C)c1c(OC)ccc([C@@H]2C(c3ccc(OC)cc3)=C[C@@]3(O)[N+H](C2)CCC3)c1",
  "O(C)c1c(OC)cc-2c([C@@H]3C(c4c-2cc(OC)c(OC)c4)=C[C@@]2(O)[N+H](C3)CCC2)c1",
  "P(=O)(O[C@H]1C(C)(C)C2(O[C@@H](C/C=C/c3nc([C@@H](CCNC(=O)[C@@H](O)[C@H](O)[C@@H]([N+H](C)C)COC)C)oc3)[C@@H](C)[C@@H](O)C2)OC1[C@@H](OC)C[C@H](O)[C@H]([C@H](O)[C@H](/C=C(\C(=C/C=C/C(=C\C#N)/C)\C)/C)C)C)([O-])[O-]",
  "O=C(C)[C@@]1(O)C[C@H](OC2OC(C)C(O)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=C([O-])C([N+H3])CC(=O)Oc1c(OC)ccc(/C=C\c2cc(OC)c(OC)c(OC)c2)c1",
  "O=[N+]([O-])c1n(CCCNc2c3c([N+](=O)[O-])cccc3[n+H]c3c2cccc3)ccn1",
  "O=[N+]([O-])c1c2c(NCCOC(=O)CC[C@@H](NC(=O)[C@@H](NC(=O)C(OC3C(O)C(CO)OC(OCc4ccccc4)C3NC(=O)C)C)C)C(=O)N)c3c([n+H]c2ccc1)cccc3",
  "O=C(CO)[C@@]1(O)C[C@H](OC2OC(C)C3OC[N+H](CN4C5C(C(C)OC(O[C@@H]6c7c([O-])c8C(=O)c9c(OC)cccc9C(=O)c8c([O-])c7C[C@@](O)(C(=O)CO)C6)C5)OC4)C3C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@](O)(CC)C[C@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "O=C(NC(C(=O)N([C@H]([C@H](OC)CC(=O)N1[C@H]([C@H](OC)[C@H](C(=O)NCCc2ccccc2)C)CCC1)[C@H](CC)C)C)C(C)C)[C@@H]([N+H](C)C)C(C)C",
  "O=C(OC(OC(=O)C)CCCC[N+H2]C1C(O)C(C)OC(O[C@H]2c3c([O-])c4C(=O)c5c(OC)cccc5C(=O)c4c([O-])c3C[C@@](O)(C(=O)CO)C2)C1)C",
  "O=[N+]([O-])c1c2c(NCCC[N+H2]C)c3c([n+H]c2ccc1)cccc3",
  "O=C([O-])[C@H]1[C@H](C)C(=O)N[C@@H](CCCNC(=[N+H2])N)C(=O)N[C@@H](/C=C/C(=C\[C@@H]([C@@H](OC)Cc2ccccc2)C)/C)[C@H](C)C(=O)N[C@@H](C(=O)[O-])CCC(=O)N(C)C(=C)C(=O)N[C@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N1",
  "O=C(OC)[C@H]1[C@](O)(C)C[C@H](OC2C(OC)C(OC)(C)C(OC)C(C)O2)c2c([O-])c3C(=O)c4c([O-])cc5[C@@]6(C)[C@H](O)[C@@H]([N+H](C)C)[C@H](O)[C@H](O6)Oc5c4C(=O)c3cc12",
  "O=[N+]([O-])c1c2c(NCCO)c3c([n+H]c2ccc1)cccc3",
  "O(C)[C@H]1[C@@H]([N+H2]C)C[C@@H]2O[C@]1(C)n1c3c4n2c2c(c4c4C(=O)NCc4c3c3c1cccc3)cccc2",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C=O)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4C[N+@H]4CC(CC)=C[C@@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)[C@H]1[C@@H](C[N+H]2CC3)C1)C",
  "O=C(CO)[C@@]1(O)C[C@H](OC2OC(C)C(O)C([N+H]3CCOCC3)C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=[N+]([O-])c1c2c(NCCCCC[N+H](C)C)c3c([n+H]c2ccc1)cccc3",
  "O=[N+]([O-])c1c2c(NCCC[N+H](C)C)c3c([n+H]c2ccc1)cccc3",
  "O=[N+]([O-])c1cc2C(=O)N(CCC[N+]#[N+][N-H])C=3c4c(C(=O)C=3c2cc1)cc(OC)cc4",
  "O(C)c1c(OC)cc2c(c3c(c4[C@@H](O)[C@H]5[N+H](Cc24)CCC5)ccc(O)c3)c1",
  "O=C1c2c([O-])ccc([O-])c2C(=O)c2c(NCC[N+H3])ccc(NCC[N+H3])c12",
  "O=[N+]([O-])c1c2c(NCCC(=O)OC)c3c([n+H]c2ccc1)cccc3",
  "O=C(N(C)C)NC1(C(O)C)C(O)(C)C(O)(COC(=O)c2c(O)cccc2C)C(Nc2cc(C(=O)C)ccc2)C1[N+H3]",
  "O=[N+]([O-])c1cc2c3c(C(=O)N(CC[N+H2]CCNCCNCCN4C(=O)c5c6c(cc([N+](=O)[O-])c5)cccc6C4=O)C(=O)c3ccc2)c1",
  "O=[N+]([O-])c1c2c(NCCCO)c3c([n+H]c2ccc1)ccc(OC)c3",
  "O=[N+]([O-])c1cc2c3c(C(=O)N(CCC[N+H](CCCNc4c5C(=O)c6c(-n7c5c(nc7)cc4)cccc6)C)C(=O)c3ccc2)c1",
  "O=C(CO)[C@@]1(O)C[C@@H](OC2OC(C)C(OC3OCCCC3)C([N+H3])C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "ClCOC(NC([N+H2]CS)C(=O)NCCNc1c2C(=O)c3c([O-])ccc([O-])c3C(=O)c2c(NCCNC(=O)C([N+H2]CS)NC(OCCl)CN(C)C)cc1)CN(C)C",
  "O=C(N)C=1[N+H2]C(C)(C)N=C2N(/N=C(\C)/C)C=NC=12",
  "FC(F)(F)c1cc2ncc(/C=N\\NC(=[N+H2])NO)c(O)c2cc1",
  "ClC1=C(N2CC[N+H](CCOCCO)CC2)C(=O)c2ncccc2C1=O",
  "O(C)c1c2C(=O)c3c([O-])c4[C@H](OC5OC(C)C(O)C([N+H]6CCOCC6)C5)C[C@@](O)(C(O)CO)Cc4c([O-])c3C(=O)c2ccc1",
  "Clc1c(C(=O)OC)cc2S(=O)(=O)N=C(N[N+H2]C)Sc2c1",
  "O=[N+]([O-])c1c2c(NCCO)c3c([n+H]c2ccc1)ccc(OC)c3",
  "O=C(C)[C@@]1(O)C[C@H](OC2OC(C)C(O)C([N+H3])C2)c2c(O)c3C(=N)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=[N+]([O-])c1c2c(NCCC[N+H3])c3c([n+H]c2ccc1)cccc3",
  "SCC([N+H3])C(=O)NCCNc1c2C(=O)c3c([O-])ccc([O-])c3C(=O)c2c(NCCNC(=O)C([N+H3])CS)cc1",
  "O=C(N[C@H]1[C@@H](C)OC(=O)[C@H](Cc2ccc(OC)cc2)N(C)C(=O)[C@H]2N(C(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)C(=O)[C@H](C(C)C)OC(=O)C[C@@H](O)[C@@H]([C@@H](CC)C)NC1=O)CCC2)[C@@H](N(C(=O)[C@H]1[N+H2]CCC1)C)CC(C)C",
  "O(C)c1cc2c3c(C)c4c(NCCC[N+H](CCCNc5c6C(=O)c7c(-n8nnc(c68)cc5)cccc7)C)nccc4c(C)c3[nH]c2cc1",
  "O=C(OC)C1[C@](O)(CC)C[C@@H](OC2OC(C)C(OC3OC(C)C(OC4C([N+H3])=CC(=O)C(C)O4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])cccc4C(=O)c3c([O-])c12",
  "O=[N+]([O-])c1c2c(NCC[N+H](CCO)CCO)c3c([n+H]c2ccc1)cccc3",
  "O=[N+]([O-])c1c2c(NCCCCCNC(=O)CCC(NC(=O)C3N(C(=O)COC4C(O)C(CO)OC(OCc5ccccc5)C4NC(=O)C)CCC3)C(=O)N)c3c([n+H]c2ccc1)cccc3",
  "O=[N+]([O-])c1c2c(NCC[N+H](CC)CC)c3c([n+H]c2ccc1)cccc3",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C=O)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@](O)(CC)C[C@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+@@H]4C[C@](O)(CC)C[C@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)[C@H]1[C@@H](C[N+H]2CC3)C1)C",
  "O(C)c1c(OC)cc2-c3c(O)c(OC)ccc3C=3[C@@H](c2c1)C[N+H]1[C@H](C=3)CCC1",
  "O=C1C([N+H]2CC2)=CC(=O)C([N+H]2CC2)=C1",
  "O=[N+]([O-])c1c2c(NCC[N+H2]CCO)c3c([n+H]c2ccc1)cccc3",
  "Clc1c(C)cc2S(=O)(=O)N=C(NNC3=[N+H]CCN3)Sc2c1",
  "O=C(N[C@H]1[C@@H](C)OC(=O)[C@H](Cc2ccc(OC)cc2)N(C)C(=O)C2N(C(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)C(=O)[C@H](C(C)C)OC(=O)C[C@H](O)[C@@H]([C@H](CC)C)NC1=O)CCC2)[C@H]([N+H](C[C@@H]1N(C(=O)C(=O)C)CCC1)C)CC(C)C",
  "Fc1c(C#CC2(O)CC[N+H2]CC2)c2/C(=C\c3c(OC)cc[nH]3)/C(=O)Nc2cc1",
  "O=C(O[C@@H]1C(C)=C2[C@@H](OC(=O)C)C(=O)[C@]3(C)[C@@H](O)C[C@@H]4[C@](OC(=O)C)([C@H]3[C@H](OC(=O)c3ccccc3)[C@@](O)(C2(C)C)C1)CO4)[C@H](O)[C@@H](NC(=O)c1cc([N+]#[N+][N-H])ccc1)c1ccccc1",
  "O=C(Nc1cc(Oc2ncc3nc(n(CC)c3c2)-c2c(N)non2)ccc1)c1ccc(OCC[N+H]2CCOCC2)cc1",
  "O=C(C[N+H](C)C)N1c2c(C(C)(C)CC1)cc(OC)c(Nc1nc(Nc3c(C(=O)N)scc3)c3c([nH]cc3)n1)c2",
  "O=C(O[C@@H]1[C@H](O)[C@H](O[C@@H]2[C@H](C)[C@H](C)[C@H](CC)O[C@@H]2O[C@H]([C@H](NC(=O)c2c(C)c(N)nc([C@@H]([N+H2]C[C@@H](N)C(=O)N)CC(=O)N)n2)C(=O)N[C@@H]([C@H](O)[C@@H](C(=O)N[C@H]([C@H](O)C)C(=O)NCCc2scc(-c3scc(C(=O)NCCCCNC(=[N+H2])N)n3)n2)C)C)c2[nH]cnc2)O[C@H](CO)[C@@H]1O)N",
  "O=C(C[N+H](C)C)N1c2c(cc(OC)c(Nc3nc(Nc4c(C(=O)N)scc4)c4c([nH]cc4)n3)c2)CCC1",
  "O=[N+]([O-])c1ccc(C[C@@H]([N+H3])[C@H](O)C(=O)N[C@H](C(=O)[O-])CC(C)C)cc1",
  "Clc1cc2[n+H]c3c(c(NCCC[N+H](CCCNc4c5C(=O)c6c(-n7c5c(nc7)cc4)cccc6)C)c2cc1)cc(OC)cc3",
  "O=C(CO)[C@@]1(O)C[C@H](OC2OC(C)C(O)C([N+H3])C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=C([O-])c1c(C)c(c(N)c(-c2nc3C(=O)C([N+H3])=C(OC)C(=O)c3cc2)n1)-c1c(O)c(OC)c(OC)cc1",
  "O=C(Oc1c(OC)ccc(/C=C\c2cc(OC)c(OC)c(OC)c2)c1)C([N+H3])CCC(=O)[O-]",
  "O=[N+]([O-])c1c2c(NCCCCCNc3c4c([N+](=O)[O-])cccc4[n+H]c4c3cccc4)c3c([n+H]c2ccc1)cccc3",
  "O(C)c1c(OC)cc(C=2C(c3cc(OC)c(OC)cc3)=C[N+H]3[C@H](C=2)CCC3)cc1O",
  "Clc1c(Cl)ccc(/C=C/2\C(=O)/C(=C\c3cc(Cl)c(Cl)cc3)/CN(C(=O)CC[N+H]3CCOCC3)C\\2)c1",
  "Clc1c(C#N)cc2S(=O)(=O)N=C(N[N+H2]C)Sc2c1",
  "O=C(O[C@@H]1[C@@](O)(C(=O)OC)[C@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@H]([C@@H]6[C@H](C4)CCC6)C3)cccc5)[C@]32[C@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "FC(F)(F)c1c(N2CC[N+H](C)CC2)ccc(Nc2nc(-c3c4n(nc3)N=CC=C4)ccn2)c1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C=O)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@](O)(CC)CC(C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)[C@H]1[C@@H](C[N+H]2CC3)C1)C",
  "O=C([O-])[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)C1N(C(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)C(NC(=O)[C@H](NC(=O)[C@H](NC(=O)C[C@H](O)CCCCCCC)CCCC[N+H3])Cc2c3c([nH]c2)cccc3)CCCC[N+H3])CC(C)C)Cc2ccccc2)CCCC[N+H3])CCCC[N+H3])CC(C)C)CCC1)CCCC[N+H3])Cc1ccccc1)CC(C)C)Cc1nc[nH]c1)CC(C)C)C)CCCC[N+H3])CCCC[N+H3])Cc1ccccc1",
  "O(C)c1c(OC)cc2c([C@@H](C[C@@H]3[C@@H](CC)C[N+H]4[C@H](c5c(cc(OC)c(OC)c5)CC4)C3)[N+H2]CC2)c1",
  "O(C)C1=C(C)C(=O)C2=C(C1=O)C(CO)N1C(C#N)C3N(C)C4C1C2[N+H]1C(OCC1)C4C3",
  "O=C1c2c([O-])ccc([O-])c2C(=O)c2c(NCC[N+H2]CCO)ccc(NCC[N+H2]CCO)c12",
  "O=C(OC1C([N+H](C)C)(C)CC(c2c([O-])c3C(=O)c4c5OC(C6(C)C(/C=C\C)O6)=CC(=O)c5c(C)cc4C(=O)c3c(C3OC(C)C(O)C([N+H](C)C)C3)c2)OC1C)C",
  "O=C(OC)C1[C@](O)(CC)C[C@@H](OC2OC(C)C(OC3OC(C)C(OC4OC(C)C(=O)CC4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])cccc4C(=O)c3c([O-])c12",
  "BrC1C([N+H3])C(O)C(C)OC1O[C@@H]1c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C[C@@](O)(C(=O)C)C1",
  "O=C(O[C@@H]1C(OC)=C[C@]23[N+H](CCc4c([C@H]12)cc1OCOc1c4)CCC3)[C@](O)(CC(=O)OC)CCCC(O)(C)C",
  "O=C(OC)C1C(O)(CC)CC(OC2OC(C)C(OC3OC(C)C4OC5OC(C)C(=O)CC5OC4C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])ccc([O-])c4C(=O)c3cc12",
  "O=[N+]([O-])c1c2c(NCCC[N+H2]C(C)C)c3c([n+H]c2ccc1)cccc3",
  "[N+H2]=C(Nc1nc(-c2ncccc2)cc2c1cccc2)c1ncccc1",
  "S(C)C=1C(=O)C=C2[C@H]([N+H2]Cc3c([N+](=O)[O-])cccc3)CCc3c(c(OC)c(OC)c(OC)c3)C2=CC=1",
  "O=C1c2c([O-])c3[C@H](OC4OC(C)C(O)C([N+H]5CCOCC5)C4)C[C@](O)(CC)[C@H](O)c3c([O-])c2C(=O)c2c1c([O-])ccc2",
  "O(C)c1cc2c3c(C)c4c(NCCC[N+H](CCCNc5c6C(=O)c7c(-n8c6c(nc8)cc5)cccc7)C)nccc4c(C)c3[nH]c2cc1",
  "O(C)c1c(OC)cc2c([C@H]3[N+H](C[C@H](CC)[C@@H](C[C@H]4[N+H2]CCc5c6c([nH]c45)ccc(O)c6)C3)CC2)c1",
  "Clc1c([C@H](Oc2c(C(=O)N)sc(-n3c4c(nc3)ccc(OC3CC[N+H](C)CC3)c4)c2)C)cccc1",
  "O=C(OC)[C@H]1[C@@](O)(CC)C[C@H](OC2OC(C)C(OC3OC(C)C(OC4OC(C)C(=O)CC4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])cccc4C(=O)c3cc12",
  "O=[N+]([O-])c1c2c(NCCCCNC(=O)CCC(NC(=O)C(NC(=O)C(OC3C(O)C(CO)OC(OCc4ccccc4)C3NC(=O)C)C)C)C(=O)N)c3c([n+H]c2ccc1)cccc3",
  "O=C(OCC)Nc1[n+H]c(N)c2N=C(c3ccccc3)CNc2c1",
  "O=C([O-])CCC1C(=O)NC(Cc2ccc(O)cc2)C(=O)NC(C(C)C)C(=O)NC(CCC[N+H3])C(=O)NC(CC(C)C)C(=O)NC(Cc2ccccc2)C(=O)N2C(C(=O)NC(Cc3ccccc3)C(=O)NC(Cc3ccccc3)C(=O)NC(CC(=O)[O-])C(=O)N1)CCC2",
  "S(C)C=1C(=O)C=C2[C@H]([N+H2]Cc3ccc([N+](=O)[O-])cc3)CCc3c(c(OC)c(OC)c(OC)c3)C2=CC=1",
  "O=C(OC)[C@H]1[C@@](O)(CC)C[C@H](OC2OC(C)C(OC3OC(C)C(OC4OC(C)C(O)C(O)C4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])ccc([O-])c4C(=O)c3cc12",
  "O=C(N[C@H]1[C@@H](C)OC(=O)[C@H](Cc2ccc(OC)cc2)N(C)C(=O)[C@H]2N(C(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)C(=O)[C@H](C(C)C)OC(=O)C[C@H](O)[C@@H]([C@H](CC)C)NC1=O)CCC2)[C@H](N(C(=O)[C@H]1[N+H2]CC=C1)C)CC(C)C",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C=O)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@](O)(CC)CC(C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "O=[N+]([O-])c1cc2c3c(C(=O)N(CCCN4CC[N+H](CCCNc5c6C(=O)c7c(-n8c6c(nc8)cc5)cccc7)CC4)C(=O)c3ccc2)c1",
  "ClCC1c2c3c(c(N)cc2N(C(=O)c2[nH]c4c(cc(OC)c(OCC[N+H](C)C)c4)c2)C1)cccc3",
  "O=C(C)[C@@]1(O)C[C@H](OC2OC(C)C(O)C([N+H3])C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4CC[N+H]4C[C@](O)(CC)CC(C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "FC(F)(F)c1cc(N2CC[N+H](CC(=C)c3ccc(C(=O)Nc4c(N)cccc4)cc3)CC2)ccc1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4C[N+@H]4CC(CC)=C[C@@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)[C@H]1[C@@H](C[N+H]2CC3)C1)C",
  "O=C(NC(C(=O)NCCC(=O)NC(C[N+H](C)C)C)(C)C)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C1N(C(=O)/C=C/C(CC)C)CC(C)C1)CC(CC(O)CC(=O)CC)C)C(O)C(C)C)(C)C)CC(C)C)CC(C)C)(C)C",
  "O(C)c1c(OC)cc2c(C(=O)N(CCC[N+H]3CC(O)CCC3)C=3c4c(C(=O)C2=3)cc2OCOc2c4)c1",
  "O=[N+]([O-])c1cc2C(=O)N(CCC[N+H3])C=3c4c(C(=O)C=3c2cc1)cccc4",
  "O=[N+]([O-])c1cc2C(=O)N(CC[N+H2]CCC[N+H2]CCN3C(=O)c4c(ccc([N+](=O)[O-])c4)C=4C(=O)c5c(cccc5)C3=4)C=3c4c(C(=O)C=3c2cc1)cccc4",
  "O=C1c2c(NCCC[N+H2]CCNCCCNc3c4C(=O)c5c(-n6nnc(c46)cc3)cccc5)ccc3nnn(c23)-c2c1cccc2",
  "O=C1N(CC[N+H2]CCC[N+H2]CCN2C(=O)c3c(cccc3)C=3C(=O)c4c(cccc4)C2=3)C=2c3c(C(=O)C=2c2c1cccc2)cccc3",
  "O=C(OC1C(O)C(OC)C(CO)OC1n1c2c3[nH]c4c(c3c3C(=O)NC(=O)c3c2c2c1cccc2)cccc4)C([N+H3])CCCC[N+H3]",
  "O=C1N(CCC[N+H2]CCCC[N+H2]CCCN2C(=O)c3c(cccc3)C3C(=O)c4c(cccc4)C23)C=2c3c(C(=O)C=2c2c1cccc2)cccc3",
  "O(C)c1c(OC)cc2c(C(=O)N(CCCN3CC[N+H2]CC3)C=3c4c(C(=O)C2=3)cc2OCOc2c4)c1",
  "Clc1c(-c2c(/C=N\\NC(=[N+H2])N)n3c(SCC3)n2)cc(Cl)s1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4C[N+H]4CC(CC)=CC(C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)C=CC[N+H]2CC3)C",
  "O=C1N(CCC[N+H2]CCCCN2C(=O)c3c(cccc3)C=3C(=O)c4c(cccc4)C2=3)C=2c3c(C(=O)C=2c2c1cccc2)cccc3",
  "O=C1/C(=C/c2ccc(C)cc2)/SC(Nc2sc(C)c(-c3c(C)sc(NC=4S/C(=C\c5ccc(C)cc5)/C(=O)[N+H]=4)n3)n2)=[N+H]1",
  "BrC=1C(=O)C(Br)=CC2(C=3C(=O)c4nc5c(c6c4c(ncc6)C=3[N+H2]CC2)cccc5)C=1",
  "Brc1ccc(N(O)C(SCC(NC(=O)CCC([N+H3])C(=O)OCC)C(=O)NCC(=O)OCC)=O)cc1",
  "FC(F)(F)c1ccc(NC(=O)NC2C(C[N+H2]CC3C(OCCCCCCCCCCCC)C4OC(C)(C)OC4O3)OC3OC(C)(C)OC23)cc1",
  "O=[N+]([O-])c1c2Nc3c(C(=O)c2c(NCCC[N+H](CCCNc2nccc4c(C)c5[nH]c6c(c5c(C)c24)cc(OC)cc6)C)cc1)cccc3",
  "O=C(O[C@]1(CC)C(=O)OCC=2C(=O)N3C(c4nc5c(c(N)c6OCOc6c5)cc4C3)=CC1=2)C[N+H3]",
  "Brc1cc2C(=O)C=3c4c(C(=O)N(CCC[N+H3])C=3c2cc1)cc([N+](=O)[O-])cc4",
  "O=C([O-])c1c(C)c(c(N)c(C2[N+H2]C=3C(=O)C([N+H3])=C(OC)C(=O)C=3CC2)n1)-c1c(O)c(OC)c(OC)cc1",
  "O=C(OC1(CC)C(=O)OCC=2C(=O)N3C(c4nc5c(cc4C3)cccc5)=CC1=2)C[N+H3]",
  "O=C(CO)[C@@]1(O)C[C@H](OC2OC(C)C3OCN4C5C(C(C)OC(O[C@@H]6c7c([O-])c8C(=O)c9c(OC)cccc9C(=O)c8c([O-])c7C[C@@](O)(C(=O)CO)C6)C5)OC[N+H](C3C2)C4)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "Clc1ccc(C23N(C(=O)CC2)c2c(N3)cc[n+H]c2)cc1",
  "O=C1C(CO)=C([N+H]2CC2)C(=O)C(C)=C1[N+H]1CC1",
  "O=[N+]([O-])c1c2C([N+H2]CCCOC(=O)CC[C@H](NC(=O)[C@@H](NC(=O)C(OC3C(O)C(CO)OC(OCc4ccccc4)C3NC(=O)C)C)C)C(=O)N)c3c(Nc2ccc1)cccc3",
  "S(C)C=1C(=O)C=C2[C@H]([N+H2]Cc3cc(C#N)ccc3)CCc3c(c(OC)c(OC)c(OC)c3)C2=CC=1",
  "O=C(O[C@@H]\\1[C@H](OC)C(=O)[C@@H](C)C[C@H](C)/C=C/C=C/C=C(/C)\[C@H](OC)C[C@H]2O[C@@](O)([C@@H](C)CC2)C(=O)C(=O)N2[C@H](C(=O)O[C@@H]([C@@H](C[C@@H]3C[C@H](OC)[C@@H](OC(=O)CC[N+H](CC)CC)CC3)C)CC(=O)[C@H](C)/C=C/1\C)CCCC2)CC[N+H](CC)CC",
  "O=C(C)C1(O)CC(OC2OC(C)C(O)C([N+H3])C2)c2c([O-])c3C(=O)c4c(C(=O)c3c([O-])c2C1)cccc4",
  "N#Cc1ccc(C2N(C(c3ccccc3)=[N+H]O2)C23CC4CC(C2)CC(C3)C4)cc1",
  "S=C(N/N=C(/C)\c1ncccc1)N1CC[N+H](C2CCCCCC2)CC1",
  "O=C(OC)C1[C@](O)(CC)C[C@@H](OC2OC(C)C(OC3OC(C)C(OC4C([N+H3])=CC(=O)C(C)O4)C(O)C3)C([N+H](C)C)C2)c2c([O-])c3C(=O)c4c([O-])cccc4C(=O)c3cc12",
  "S=C(N/N=C(/C)\c1ncccc1)N1CC[N+H](CC#C)CC1",
  "O=C(N[C@H]1[C@@H](C)OC(=O)[C@H](Cc2ccc(OC)cc2)N(C)C(=O)[C@H]2N(C(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)C(=O)[C@H](C(C)C)OC(=O)C[C@@H](O)[C@@H]([C@@H](CC)C)NC1=O)CCC2)[C@@H](N(C(=O)[C@H]1[N+H2]C[C@@H](O)C1)C)CC(C)C",
  "O([C@@H](C[N+H3])c1ccccc1)c1nc(C#CC(O)(C)C)c2nc(n(CC)c2c1)-c1c(N)non1",
  "O=C(CO)[C@@]1(O)C[C@H](OC2O[C@@H](C)[C@@H](O)[C@@H]([N+H3])C2)c2c([O-])c3C(=O)c4c(OC)cccc4C(=O)c3c([O-])c2C1",
  "O=C(OC(OC(=O)C)COCC[N+H2]C1C(O)C(C)OC(O[C@H]2c3c([O-])c4C(=O)c5c(OC)cccc5C(=O)c4c([O-])c3C[C@@](O)(C(=O)CO)C2)C1)C",
  "S(C)C=1C(=O)C=C2[C@H]([N+H2]Cc3ccc(C#N)cc3)CCc3c(c(OC)c(OC)c(OC)c3)C2=CC=1",
  "Clc1cc(Cl)cc(S(=O)(=O)NC(CC(=O)NCC[N+H]2CCCC2)C2C(OCCCCCCCCCCCC)C3OC(C)(C)OC3O2)c1",
  "Fc1cc(Nc2nc(Nc3c(OC)cc(C4CC[N+H](CCC)CC4)cc3)nc3[nH]ccc23)c(C(=O)N)cc1",
  "O(CC[N+H](C)C)c1ccc(-c2[nH]c(c(-c3ccncc3)n2)-c2cc3c(/C(=N\O)/CC3)cc2)cc1",
  "Fc1c(c(F)cc2c1N(C1CC1)C=C1C(=O)N(C3CCC([N+H](C)C)CC3)N=C21)-c1cc(C)nc(C)c1",
  "O(C)c1ccc(C[C@@H]2N(C)C(=O)[C@H]3N(C(=O)[C@H](CC(C)C)NC(=O)[C@@H](C)C(=O)[C@H](C(C)C)OC(=O)C[C@H](O)[C@@H]([C@H](CC)C)NC(=O)[C@@H]([N+H2][C@](O)([C@@H](N)CC(C)C)C)[C@@H](C)OC2=O)CCC3)cc1",
  "O=C(O[C@H]1[C@](O)(C(=O)OC)[C@@H]2N(C)c3c(cc(c(OC)c3)[C@]3(C(=O)OC)c4[nH]c5c(c4C[N+@@H]4C[C@](O)(CC)C[C@@H](C3)C4)cccc5)[C@@]32[C@@H]2[C@@]1(CC)[C@H]1[C@@H](C[N+H]2CC3)C1)C",
  "O=[N+]([O-])c1c2c(NCCC[N+H](CC)CC)c3c([n+H]c2ccc1)ccc(OC)c3",
  "O=C(O[C@]1(CC)C(=O)OCC=2C(=O)N3C(c4nc5c(cc4C3)cccc5)=CC1=2)C[N+H](C)C",
  "O=[N+]([O-])c1c2c(NCCC[N+H2]CCCCCC)c3c([n+H]c2ccc1)cccc3"
]

def load_prot_dict(protein_list, prot_desc_path, sequence_field, 
  phospho_field):
  if re.search('davis', prot_desc_path, re.I):
    source = 'davis'
  elif re.search('metz', prot_desc_path, re.I):
    source = 'metz'
  elif re.search('kiba', prot_desc_path, re.I):
    source = 'kiba'
  elif re.search('toxcast', prot_desc_path, re.I):
    source = 'toxcast'

  df = pd.read_csv(prot_desc_path, index_col=0)
  for row in df.itertuples():
    descriptor = row[2:]
    descriptor = np.array(descriptor)
    descriptor = np.reshape(descriptor, (1, len(descriptor)))
    pair = (source, row[0])      
    sequence = row[sequence_field]
    phosphorylated = row[phospho_field]
    protein = Protein(row[0], source, (phosphorylated, sequence))
    if protein not in set(protein_list):
      protein_list.append(protein)  
    
def produce_dataset(dataset_used='toxcast'):
  assert dataset_used in ['toxcast', 'kiba']
  df = pd.read_csv('NCI60_bio.csv', header = 0, index_col=2)
  #df = df.head(60000)
  molList = list(df.index)
  molList = [mol for mol in molList if mol==mol]
  assert len(df) == len(molList)
  selected_mol_set = set()
  selected_mol_list = []
  mols_to_skip = 0
  mols_to_choose = 50000
  GIarray = np.asarray(df.iloc[:, 5])
  sorted_indices = np.argsort(GIarray)
  for i in sorted_indices:
    smiles = molList[i]
    if smiles not in selected_mol_set:
      selected_mol_set.add(smiles)
      selected_mol_list.append(smiles)
      if len(selected_mol_set) >= mols_to_skip + mols_to_choose:
        break

  loading_functions = {
    'davis': dcCustom.molnet.load_davis,
    'metz': dcCustom.molnet.load_metz,
    'kiba': dcCustom.molnet.load_kiba,
    'toxcast': dcCustom.molnet.load_toxcast,
    'all_kinase': dcCustom.molnet.load_kinases,
    'tc_kinase':dcCustom.molnet.load_tc_kinases,
    'tc_full_kinase': dcCustom.molnet.load_tc_full_kinases
  }  

  selected_mol_list = selected_mol_list[mols_to_skip:]
  tasks, _, _ = loading_functions[dataset_used](featurizer="ECFP", currdir="../")
  
  prot_desc_path_list = ['../davis_data/prot_desc.csv', '../metz_data/prot_desc.csv', 
    '../KIBA_data/prot_desc.csv', '../full_toxcast/prot_desc.csv']
  #prot_desc_path_list = ['../metz_data/prot_desc.csv']
  prot_list = []

  for path in prot_desc_path_list:
    load_prot_dict(prot_list, path, 1, 2)

  #prot_list = prot_list[:1]
  start_writing = time.time()
  suffix = {
    'toxcast': '_tc',
    'kiba': '_kiba'
  }
  fname = 'restructured_toy' + suffix[dataset_used] + '.csv'
  with open(fname, 'w', newline='') as csvfile:
    fieldnames = tasks + ['smiles', 'proteinName', 'protein_dataset']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for mol in selected_mol_list:
      if mol in set(invalid_mols):
        continue
      for prot in prot_list:
        prot_source_and_name = prot.get_name()
        out_line = {'smiles': mol, 'proteinName': prot_source_and_name[1], 
          'protein_dataset': prot_source_and_name[0]}
        line_values = dict(zip(tasks, [0]*len(tasks)))
        out_line.update(line_values)
        writer.writerow(out_line)

  end_writing = time.time()
  print("Time spent in writing: ", end_writing - start_writing)

def synthesize_ranking(prediction_file, output_file, direction=True, dataset_used='toxcast'):
  assert dataset_used in ['toxcast', 'kiba']
  csv_data = {
    'toxcast': '../full_toxcast/restructured.csv',
    'kiba': '../KIBA_data/restructured_unique.csv'
  }
  df = pd.read_csv(csv_data[dataset_used], header=0, index_col=False)
  if dataset_used == 'toxcast':
    tasks, _, _ = dcCustom.molnet.load_toxcast(featurizer="ECFP", currdir="../")
  elif dataset_used == 'kiba':
    tasks, _, _ = dcCustom.molnet.load_kiba(featurizer="ECFP", currdir="../", cross_validation=True, 
      split_warm=True, filter_threshold=6)
  datapoints = []
  for task in tasks:
    task_vector = np.asarray(df.loc[:, task])
    datapoints.append(np.count_nonzero(~np.isnan(task_vector)))
  datapoints = np.array(datapoints)
  fractions = []
  for i in range(len(datapoints)):
    fractions.append(datapoints[i]/datapoints.sum())

  preds_df = pd.read_csv(prediction_file, header=0, index_col=False)
  compounds = preds_df.loc[:, 'Compound']
  prot_names = preds_df.loc[:, 'proteinName']
  prot_sources = preds_df.loc[:, 'protein_dataset']
  composite_preds = np.zeros_like(preds_df.loc[:, tasks[0]])
  for j in range(len(tasks)):
    task = tasks[j]
    pred_task = preds_df.loc[:, task] * fractions[j]
    composite_preds += pred_task
  if direction:
    neg_composite_preds = -1 * composite_preds
    sorted_indices = neg_composite_preds.argsort()
  else:
    sorted_indices = composite_preds.argsort()
  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'proteinName', 'protein_dataset', 'synthesized_score']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in sorted_indices:
      out_line = {'smiles': compounds[i], 'proteinName': prot_names[i], 
        'protein_dataset': prot_sources[i], 'synthesized_score': composite_preds[i]}
      writer.writerow(out_line)  

def compare(file_1, file_2, cutoff=None):
  df_1 = pd.read_csv(file_1, header=0, index_col=False)
  df_2 = pd.read_csv(file_2, header=0, index_col=False)
  if cutoff is not None:
    df_1 = df_1.head(cutoff)
    df_2 = df_2.head(cutoff)
  pred_triplets_set_1 = set()
  pred_triplets_set_2 = set()
  for row in df_1.itertuples():
    pred_triplets_set_1.add((row[1], row[2], row[3]))
  for row in df_2.itertuples():
    pred_triplets_set_2.add((row[1], row[2], row[3]))
  intersec = pred_triplets_set_1.intersection(pred_triplets_set_2)
  print(len(intersec))
  #pdb.set_trace()

def get_invalid_smiles(out_file='invalid_smiles.csv'):
  err_log = open('../logs/error.log', 'r')
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['invalid_smiles']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for line in err_log:
      raw_smiles = re.search(r"'.+'", line).group(0)
      smiles=raw_smiles[1:-1]
      out_line = {'invalid_smiles': smiles}
      writer.writerow(out_line)
  err_log.close()

if __name__ == "__main__":
  dataset = 'toxcast'
  #dataset = 'kiba'
  #produce_dataset(dataset_used=dataset)
  #synthesize_ranking('preds_kiba_graphconv.csv', 'ordered_kiba_gc.csv',   
  # synthesize_ranking('preds_tc_graphconv.csv', 'synthesized_values_gc.csv', 
  #   direction=True, dataset_used=dataset)
  #compare('ordered_kiba.csv', 'synthesized_values.csv', cutoff=2000)
  get_invalid_smiles(out_file = 'invalid_smiles.csv')