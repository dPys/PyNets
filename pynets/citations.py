from pynets.__about__ import __version__
from datetime import datetime

PYNETS = """
@CONFERENCE{dPys,
    title = {PyNets v{version}: A Reproducible Workflow for Structural and
    Functional Connectome Ensemble Learning'},
    author = {Pisner, D.},
    publisher = {Poster session presented at: Annual Meeting of the Organization
    for Human Brain Mapping},
    url = {https://github.com/dPys/PyNets},
    year = {2020},
    month = {June}
}
""".format(
    version=__version__, datestamp=datetime.utcnow().strftime("%Y-%m-%d")
).strip()

BRANDES2001 = """
@article{brandes2008variants,
  title={On variants of shortest-path betweenness centrality and their generic computation},
  author={Brandes, Ulrik},
  journal={Social Networks},
  volume={30},
  number={2},
  pages={136--145},
  year={2008},
  publisher={Elsevier}
}
""".strip()

KINTALI2008 = """
@article{kintali2008betweenness,
  title={Betweenness centrality: Algorithms and lower bounds},
  author={Kintali, Shiva},
  journal={arXiv preprint arXiv:0809.1906},
  year={2008}
}
""".strip()

WATTS1998 = """
@article{watts1998collective,
  title={Collective dynamics of `small-world' networks},
  author={Watts, Duncan J and Strogatz, Steven H},
  journal={nature},
  volume={393},
  number={6684},
  pages={440},
  year={1998},
  publisher={Nature Publishing Group}
}
""".strip()

ONNELA2005 = """
@article{onnela2005intensity,
  title={Intensity and coherence of motifs in weighted complex networks},
  author={Onnela, Jukka-Pekka and Saram{\\"a}ki, Jari and Kert{\\'e}sz, J{\\'a}nos and Kaski, Kimmo},
  journal={Physical Review E},
  volume={71},
  number={6},
  pages={065103},
  year={2005},
  publisher={APS}
}
""".strip()

LANCICHINETTI2012 = """
@article{lancichinetti2012consensus,
  title={Consensus clustering in complex networks},
  author={Lancichinetti, Andrea and Fortunato, Santo},
  journal={Scientific reports},
  volume={2},
  pages={336},
  year={2012},
  publisher={Nature Publishing Group}
 }
""".strip()

RUBINOV2011 = """
@article{rubinov2011weight,
  title={Weight-conserving characterization of complex functional brain networks},
  author={Rubinov, Mikail and Sporns, Olaf},
  journal={Neuroimage},
  volume={56},
  number={4},
  pages={2068--2079},
  year={2011},
  publisher={Elsevier}
}
""".strip()

LATORA2001 = """
@article{latora2001efficient,
  title={Efficient behavior of small-world networks},
  author={Latora, Vito and Marchiori, Massimo},
  journal={Physical review letters},
  volume={87},
  number={19},
  pages={198701},
  year={2001},
  publisher={APS}
}
""".strip()

RUBINOV2010 = """
@article{rubinov2010complex,
  title={Complex network measures of brain connectivity: uses and interpretations},
  author={Rubinov, Mikail and Sporns, Olaf},
  journal={Neuroimage},
  volume={52},
  number={3},
  pages={1059--1069},
  year={2010},
  publisher={Elsevier}
}
""".strip()

NEWMAN2016 = """
@article{newman2016mathematics,
  title={Mathematics of networks},
  author={Newman, Mark EJ},
  journal={The new Palgrave dictionary of economics},
  pages={1--8},
  year={2016},
  publisher={Springer}
}
""".strip()

HONEY2007 = """
@article{honey2007network,
  title={Network structure of cerebral cortex shapes functional connectivity on multiple time scales},
  author={Honey, Christopher J and K{\\"o}tter, Rolf and Breakspear, Michael and Sporns, Olaf},
  journal={Proceedings of the National Academy of Sciences},
  volume={104},
  number={24},
  pages={10240--10245},
  year={2007},
  publisher={National Acad Sciences}
}
"""

SPORNS2004 = """
@article{sporns2004small,
  title={The small world of the cerebral cortex},
  author={Sporns, Olaf and Zwi, Jonathan D},
  journal={Neuroinformatics},
  volume={2},
  number={2},
  pages={145--162},
  year={2004},
  publisher={Springer}
}
""".strip()

REICHARDT2006 = """
@article{reichardt2006statistical,
  title={Statistical mechanics of community detection},
  author={Reichardt, J{\\"o}rg and Bornholdt, Stefan},
  journal={Physical Review E},
  volume={74},
  number={1},
  pages={016110},
  year={2006},
  publisher={APS}
}
""".strip()

GOOD2010 = """
@article{good2010performance,
  title={Performance of modularity maximization in practical contexts},
  author={Good, Benjamin H and De Montjoye, Yves-Alexandre and Clauset, Aaron},
  journal={Physical Review E},
  volume={81},
  number={4},
  pages={046106},
  year={2010},
  publisher={APS}
}
""".strip()

SUN2008 = """
@article{sun2009improved,
  title={Improved community structure detection using a modified fine-tuning strategy},
  author={Sun, Yudong and Danila, Bogdan and Josi{\'c}, K and Bassler, Kevin E},
  journal={EPL (Europhysics Letters)},
  volume={86},
  number={2},
  pages={28004},
  year={2009},
  publisher={IOP Publishing}
}
""".strip()

BLONDEL2008 = """
@article{blondel2008fast,
  title={Fast unfolding of communities in large networks},
  author={Blondel, Vincent D and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  journal={Journal of statistical mechanics: theory and experiment},
  volume={2008},
  number={10},
  pages={P10008},
  year={2008},
  publisher={IOP Publishing}
}
""".strip()

MEILA2007 = """
@article{meila2007comparing,
  title={Comparing clusterings -- an information based distance},
  author={Meil{\\={a}}, Marina},
  journal={Journal of multivariate analysis},
  volume={98},
  number={5},
  pages={873--895},
  year={2007},
  publisher={Elsevier}
}
""".strip()

BASSETT2010 = """
@article{bassett2010efficient,
  title={Efficient physical embedding of topologically complex information processing networks in brains and computer circuits},
  author={Bassett, Danielle S and Greenfield, Daniel L and Meyer-Lindenberg, Andreas and Weinberger, Daniel R and Moore, Simon W and Bullmore, Edward T},
  journal={PLoS computational biology},
  volume={6},
  number={4},
  pages={e1000748},
  year={2010},
  publisher={Public Library of Science}
}
""".strip()

COLIZZA2006 = """
@article{colizza2006detecting,
  title={Detecting rich-club ordering in complex networks},
  author={Colizza, Vittoria and Flammini, Alessandro and Serrano, M Angeles and Vespignani, Alessandro},
  journal={Nature physics},
  volume={2},
  number={2},
  pages={110},
  year={2006},
  publisher={Nature Publishing Group}
}
""".strip()

OPSAHL2008 = """
@article{opsahl2008prominence,
  title={Prominence and control: the weighted rich-club effect},
  author={Opsahl, Tore and Colizza, Vittoria and Panzarasa, Pietro and Ramasco, Jose J},
  journal={Physical review letters},
  volume={101},
  number={16},
  pages={168702},
  year={2008},
  publisher={APS}
}
""".strip()

HEUVEL2011 = """
@article{van2011rich,
  title={Rich-club organization of the human connectome},
  author={Van Den Heuvel, Martijn P and Sporns, Olaf},
  journal={Journal of Neuroscience},
  volume={31},
  number={44},
  pages={15775--15786},
  year={2011},
  publisher={Soc Neuroscience}
}
""".strip()

ESTRADA2005 = """
@article{estrada2005subgraph,
  title={Subgraph centrality in complex networks},
  author={Estrada, Ernesto and Rodriguez-Velazquez, Juan A},
  journal={Physical Review E},
  volume={71},
  number={5},
  pages={056103},
  year={2005},
  publisher={APS}
}
""".strip()

ESTRADA2010 = """
@article{estrada2010network,
  title={Network properties revealed through matrix functions},
  author={Estrada, Ernesto and Higham, Desmond J},
  journal={SIAM review},
  volume={52},
  number={4},
  pages={696--714},
  year={2010},
  publisher={SIAM}
}
""".strip()

HUMPHRIES2008 = """
@article{humphries2008network,
  title={Network `small-world-ness': a quantitative method for determining canonical network equivalence},
  author={Humphries, Mark D and Gurney, Kevin},
  journal={PloS one},
  volume={3},
  number={4},
  pages={e0002051},
  year={2008},
  publisher={Public Library of Science}
}
""".strip()


BETZEL2016 = """
@article{betzel2016generative,
  title={Generative models of the human connectome},
  author={Betzel, Richard F and Avena-Koenigsberger, Andrea and Go{\\~n}i, Joaqu{\\'\\i}n and He, Ye and De Reus, Marcel A and Griffa, Alessandra and V{\\'e}rtes, Petra E and Mi{\\v{s}}ic, Bratislav and Thiran, Jean-Philippe and Hagmann, Patric and others},
  journal={Neuroimage},
  volume={124},
  pages={1054--1064},
  year={2016},
  publisher={Elsevier}
}
""".strip()

VAROQUAUX2013 = """
@article{Varoquaux2013,
  arxivId = {1304.3880},
  author = {Varoquaux, Ga{\"{e}}l and Craddock, R. Cameron},
  doi = {10.1016/j.neuroimage.2013.04.007},
  journal = {NeuroImage},
  pages = {405--415},
  pmid = {23583357},
  title = {{Learning and comparing functional connectomes across subjects}},
  volume = {80},
  year = {2013}
}
""".strip()

YUEN2019 = """
@article{Yuen2019,
  author = {Yuen, Nicole H. and Osachoff, Nathaniel and Chen, J. Jean},
  doi = {10.3389/fnins.2019.00900},
  journal = {Frontiers in Neuroscience},
  title = {{Intrinsic Frequencies of the Resting-State fMRI Signal: The Frequency Dependence of Functional Connectivity and the Effect of Mode Mixing}},
  year = {2019}
}
""".strip()

CRADDOCK2013 = """
@article{Craddock2013a,
  author = {Craddock, R Cameron and James, G Andrew and Iii, Paul E Holtzheimer and Hu, Xiaoping P and Mayberg, Helen S},
  journal = {Human brain mapping},
  number = {8},
  pages = {1914--1928},
  title = {{A whole brain fMRI atlas spatial Generated via Spatially Constrained Spectral Clustering{\_} Craddock, James 2011 .pdf}},
  volume = {33},
  year = {2013}
}
""".strip()

XIANG2020 = """
@article{Xiang2020,
  author = {Xiang, Yu Tao and Yang, Yuan and Li, Wen and Zhang, Ling and Zhang, Qinge and Cheung, Teris and Ng, Chee H.},
  booktitle = {The Lancet Psychiatry},
  doi = {10.1016/S2215-0366(20)30046-8},
  title = {{Timely mental health care for the 2019 novel coronavirus outbreak is urgently needed}},
  year = {2020}
}
""".strip()

BASSETT2017 = """
@article{Bassett2017,
  author = {Bassett, Danielle S. and Sporns, Olaf},
  booktitle = {Nature Neuroscience},
  doi = {10.1038/nn.4502},
  title = {{Network neuroscience}},
  year = {2017}
}
""".strip()

SPORNS2012 = """
@article{Sporns2012,
  author = {Sporns, Olaf},
  booktitle = {NeuroImage},
  doi = {10.1016/j.neuroimage.2011.08.085},
  title = {{From simple graphs to the connectome: Networks in neuroimaging}},
  year = {2012}
}
""".strip()

BULLMORE2009 = """
@article{Bullmore2009,
  author = {Bullmore, Ed and Sporns, Olaf},
  booktitle = {Nature Reviews Neuroscience},
  doi = {10.1038/nrn2575},
  title = {{Complex brain networks: Graph theoretical analysis of structural and functional systems}},
  year = {2009}
}
""".strip()

GARCIAGARCIA2018 = """
@article{Garcia-Garcia2018,
  author = {Garcia-Garcia, Manuel and Nikolaidis, Aki and Bellec, Pierre and Craddock, R. Cameron and Cheung, Brian and Castellanos, Francisco X. and Milham, Michael P.},
  doi = {10.1016/j.neuroimage.2017.07.029},
  journal = {NeuroImage},
  pages = {68--82},
  title = {{Detecting stable individual differences in the functional organization of the human basal ganglia}},
  Volume = {170}
  year = {2018}
  publisher = {Elsevier}
}
""".strip()

GREENE2017 = """
@article{Greene2017,
  author = {Greene, Clint and Cieslak, Matt and Grafton, Scott T.},
  doi = {10.1162/netn_a_00035},
  journal = {Network Neuroscience},
  title = {{Effect of different spatial normalization approaches on tractography and structural brain networks}},
  year = {2017}
}
""".strip()

SCHAEFER2018 = """
@article{Schaefer2018,
    author = {Schaefer, Alexander and Kong, Ru and Gordon, Evan M and Laumann, Timothy ) and Zuo, Xi-Nian and Holmes, Avram J and Eickhoff, Simon B and Yeo, BT Thomas},
    journal = {Cerebral Cortex},
    number = {9},
    pages = {3095--3114}},
    title = {{Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI}},
    volume = {28},
    year = {2018},
    publisher = {Oxford University Press}
}
""".strip()

THOMAS2011 = """
@article{Thomas2011,
    author = {Thomas Yeo, BT and Krienen, Fenna M and Sepulcre, Jorge and Sabuncu, Mert R and Lashkari, Danial and Hollinshead, Marisa and Roffman, Joshua L and Smoller, Jordan W and Z{\"o}llei, Lilla and Polimeni, Jonathan R and others},
    journal = {Journal of Neurophysiology},
    number = {3},
    pages = {1125--1165},
    title = {{The organization of the human cerebral cortex estimated by intrinsic functional connectivity}},
    volume = {106},
    year = {2011},
    publisher = {American Physiological Society Bethesda, MD}
}
""".strip()

TZOURIO2002 = """
@article{Tzourio2002,
    author = {Tzourio-Mazoyer, Nathalie and Landeau, Brigitte and Papathanassiou, Dimitri and Crivello, Fabrice and Etard, Olivier and Delcroix, Nicolas and Mazoyer, Bernard and Joliot, Marc},
    journal = {Neuroimage},
    number = {1},
    pages = {273--289},
    title = {{Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain}},
    volume = {15},
    year = {2002},
    publisher = {Academic Press}
}
""".strip()

YARKONI2011 = """
@article{Yarkoni2011,
    author = {Yarkoni, Tal and Poldrack, Russell A and Nichols, Thomas E and Van Essen, David C and Wager, Tor D},
    journal = {Nature methods},
    number = {8},
    pages = {665},
    title = {{Large-scale automated synthesis of human functional neuroimaging data}},
    volume = {8},
    year = {2011},
    publisher = {Nature Publishing Group}
}
""".strip()

TAUSCZIK2010 = """
@article{Tausczik2010,
    author = {Tausczik, Yla R and Pennebaker, James W},
    journal = {Journal of language and social psychology},
    number = {1},
    pages = {24--54},
    title = {{The psychological meaning of words: LIWC and computerized text analysis methods}},
    volume = {29},
    year = {2010},
    publisher = {Sage Publications Sage CA: Los Angeles, CA}
}
""".strip()

VANWIJK2010 = """
@article{VanWijk2010,
    author = {Van Wijk, Bernadette CM and Stam, Cornelis J and Daffertshofer, Andreas},
    journal = {PloS one},
    number = {10},
    title = {{Comparing brain networks of different size and connectivity density using graph theory}},
    volume = {5},
    year = {2010},
    publisher = {Public Library of Science}
}
""".strip()

SERRANO2009 = """
@article{Serrano2009,
    author = {Serrano, M {\'A}ngeles and Bogun{\'a}, Mari{\'a}n and Vespignani, Alessandro},
    journal = {Proceedings of the national academy of sciences},
    number = {16},
    pages = {6483--6488},
    title = {{Extracting the multiscale backbone of complex weighted networks}},
    volume = {106},
    year = {2009},
    publisher = {National Acad Sciences}
}
""".strip()

ALEXANDER2010 = """
@article{Alexander2010,
    author = {Alexander-Bloch, Aaron F and Gogtay, Nitin and Meunier, David and Birn, Rasmus and Clasen, Liv and Lalonde, Francois and Lenroot, Rhoshel and Giedd, Jay and Bullmore, Edward T},
    journal = {Frontiers in systems neuroscience},
    pages = {147},
    title = {{Disrupted modularity and local connectivity of brain functional networks in childhood-onset schizophrenia}},
    volume = {4},
    year = {2010},
    publisher = {Frontiers}
}
""".strip()

TEWARIE2015 = """
@article{Tewarie2015,
    author = {Tewarie, Prejaas and van Dellen, Edwin and Hillebrand, Arjan and Stam, Cornelis J},
    journal = {Neuroimage},
    pages = {177--188},
    title = {{The minimum spanning tree: an unbiased method for brain network analysis}},
    volume = {104},
    year = {2015},
    publisher = {Elsevier}
}
""".strip()

CHEN2019 = """
@article{Chen2019,
    author = {Chen, David Qixiang and Dell’Acqua, Flavio and Rokem, Ariel and Garyfallidis, Eleftherios and Hayes, David J and Zhong, Jidan and Hodaie, Mojgan},
    journal = {BioRxiv},
    pages = {864108},
    title = {{Diffusion Weighted Image Co-registration: Investigation of Best Practices}},
    year = {2019},
    publisher = {Cold Spring Harbor Laboratory}
}
""".strip()

BASSER1994 = """
@article{Basser1994,
    author = {Basser, Peter J and Mattiello, James and LeBihan, Denis},
    journal = {Biophysical journal},
    number = {1},
    pages = {259--267},
    title = {{MR diffusion tensor spectroscopy and imaging}},
    volume = {66},
    year = {1994},
    publisher = {Elsevier}
}
""".strip()

PAJEVIC1999 = """
@article{Pajevic1999,
    author = {Pajevic, Sinisa and Pierpaoli, Carlo},
    journal = {Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine},
    number = {3},
    pages = {526--540},
    title = {{Color schemes to represent the orientation of anisotropic tissues from diffusion tensor data: application to white matter fiber tract mapping in the human brain}},
    volume = {42},
    year = {1999},
    publisher = {Wiley Online Library}
  }
""".strip()

AGANJ2009 = """
@article{Aganj2009,
    author = {Aganj, Iman and Lenglet, Christophe and Sapiro, Guillermo},
    booktitle = {2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro},
    pages = {1398--1401},
    title = {{ODF reconstruction in q-ball imaging with solid angle consideration}},
    year = {2009},
    organization = {IEEE}
}
""".strip()

TOURNIER2007 = """
@article{Tournier2007,
    author = {Tournier, J-Donald and Calamante, Fernando and Connelly, Alan},
    journal = {Neuroimage},
    number = {4},
    pages = {1459--1472},
    title = {{Robust determination of the fibre orientation distribution in diffusion MRI: non-negativity constrained super-resolved spherical deconvolution}},
    volume = {35},
    year = {2007},
    publisher = {Elsevier}
}
""".strip()

DESCOTEAUX2008 = """
@article{Descoteaux2008,
    author = {Descoteaux, Maxime and Deriche, Rachid and Knosche, Thomas R and Anwander, Alfred},
    journal = {IEEE transactions on medical imaging},
    number = {2},
    pages = {269--286},
    title = {{Deterministic and probabilistic tractography based on complex fibre orientation distributions}},
    volume = {28},
    year = {2008},
    publisher = {IEEE}
}
""".strip()

COTE2013 = """
@article{Cote2013,
    author = {C{\^o}t{\'e}, Marc-Alexandre and Girard, Gabriel and Bor{\'e}, Arnaud and Garyfallidis, Eleftherios and Houde, Jean-Christophe and Descoteaux, Maxime},
    journal = {Medical image analysis},
    number = {7},
    pages = {844--857},
    title = {{Tractometer: towards validation of tractography pipelines}},
    volume = {17},
    year = {2013},
    publisher = {Elsevier}
}
""".strip()

TOURNIER2012 = """
@article{Tournier2012,
    author = {Tournier, J-Donald and Calamante, Fernando and Connelly, Alan},
    journal = {International journal of imaging systems and technology},
    number = {1},
    pages = {53--66},
    title = {{MRtrix: diffusion tractography in crossing fiber regions}},
    volume = {22},
    year = {2012},
    publisher = {Wiley Online Library}
}
""".strip()

ROKEM2015 = """
@article{Rokem2015,
    author = {Rokem, Ariel and Yeatman, Jason D and Pestilli, Franco and Kay, Kendrick N and Mezer, Aviv and Van Der Walt, Stefan and Wandell, Brian A},
    journal = {PloS one},
    number = {4},
    title = {{Evaluating the accuracy of diffusion MRI models in white matter}},
    volume = {10},
    year = {2015},
    publisher = {Public Library of Science}
}
""".strip()

SPORNS2005 = """
@article{Sporns2005,
    author = {Sporns, Olaf and Tononi, Giulio and K{\"o}tter, Rolf},
    journal = {PLoS computational biology},
    number = {4},
    title = {The human connectome: a structural description of the human brain},
    volume = {1},
    year = {2005},
    publisher = {Public Library of Science}
}
""".strip()

SOTIROPOULOS2019 = """
@article{Sotiropoulos2019,
    author = {Sotiropoulos, Stamatios N and Zalesky, Andrew},
    journal = {NMR in Biomedicine},
    number = {4},
    pages = {e3752},
    volume = {32},
    title = {{Building connectomes using diffusion MRI: why, how and but}},
    year = {2019},
    publisher = {Wiley Online Library}
}
""".strip()

CHUNG2017 = """
@article{Chung2017,
    author = {Chung, Moo K and Hanson, Jamie L and Adluru, Nagesh and Alexander, Andrew L and Davidson, Richard J and Pollak, Seth D},
    journal = {Brain connectivity},
    number = {6},
    pages = {331--346},
    title = {{Integrative structural brain network analysis in diffusion tensor imaging}},
    volume = {7},
    year = {2017},
    publisher = {Mary Ann Liebert, Inc. 140 Huguenot Street, 3rd Floor New Rochelle, NY 10801 USA}
}
""".strip()

SOARES2013 = """
@article{Soares2013,
    author = {Soares, Jose and Marques, Paulo and Alves, Victor and Sousa, Nuno},
    journal = {Frontiers in neuroscience},
    pages = {31},
    title = {{A hitchhiker's guide to diffusion tensor imaging}},
    volume = {7},
    year = {2013},
    publisher = {Frontiers}
}
""".strip()

ZHANG2001 = """
@article{Zhang2001,
    author = {Zhang, Yongyue and Brady, Michael and Smith, Stephen},
    journal = {IEEE transactions on medical imaging},
    number = {1},
    pages = {45--57},
    title = {{Segmentation of brain MR images through a hidden Markov random field model and the expectation-maximization algorithm}},
    volume = {20},
    year = {2001},
    publisher = {Ieee}
}
""".strip()

AVANTS2011 = """
@article{Avants2011,
    author = {Avants, Brian B and Tustison, Nicholas J and Wu, Jue and Cook, Philip A and Gee, James C},
    journal = {Neuroinformatics},
    number = {4},
    pages = {381--400},
    title = {{An open source multivariate framework for n-tissue segmentation with evaluation on public data}},
    volume = {9},
    year = {2011},
    publisher = {Springer}
}
""".strip()

TAKEMURA2016 = """
@article{Takemura2016,
    author = {Takemura, Hiromasa and Caiafa, Cesar F and Wandell, Brian A and Pestilli, Franco},
    journal = {PLoS computational biology},
    number = {2},
    title = {{Ensemble tractography}},
    volume = {12},
    year = {2016},
    publisher = {Public Library of Science}
}
""".strip()

SHI2000 = """
@article{Shi2000,
    author = {Shi, Jianbo and Malik, Jitendra},
    journal = {IEEE Transactions on pattern analysis and machine intelligence},
    number = {8},
    pages = {888--905},
    volume = {22},
    title = {{Normalized cuts and image segmentation}},
    year = {2000},
    publisher = {Ieee}
}
""".strip()

CRADDOCK2012 = """
@article{Craddock2012,
    author = {Craddock, R Cameron and James, G Andrew and Holtzheimer III, Paul E and Hu, Xiaoping P and Mayberg, Helen S},
    journal = {Human brain mapping},
    number = {8},
    pages = {1914--1928},
    title = {{A whole brain fMRI atlas generated via spatially constrained spectral clustering}},
    volume = {33},
    year = {2012},
    publisher = {Wiley Online Library}
}
""".strip()

THIRION2014 = """
@article{Thirion2014,
    author = {Thirion, Bertrand and Varoquaux, Ga{\"e}l and Dohmatob, Elvis and Poline, Jean-Baptiste},
    journal = {Frontiers in neuroscience},
    pages = {167},
    volume = {8},
    title = {{Which fMRI clustering gives good brain parcellations?}},
    year = {2014},
    publisher = {Frontiers}
}
""".strip()

BELLEC2010 = """
@article{Bellec2010,
    author = {Bellec, Pierre and Rosa-Neto, Pedro and Lyttelton, Oliver C and Benali, Habib and Evans, Alan C},
    journal = {Neuroimage},
    number = {3},
    pages = {1126--1139},
    title = {{Multi-level bootstrap analysis of stable clusters in resting-state fMRI}},
    volume = {51},
    year = {2010},
    publisher = {Elsevier}
    }
""".strip()

BELLEC2008 = """
@article{Bellec2008,
    author = {Bellec, Pierre and Marrelec, Guillaume and Benali, Habib},
    journal = {Statistica Sinica},
    pages = {1253--1268},
    title = {{A bootstrap test to investigate changes in brain connectivity for functional MRI}},
    year={2008},
    publisher={JSTOR}
}
""".strip()

GREENE2018 = """
@article{Greene2018,
    author = {Greene, Clint and Cieslak, Matt and Grafton, Scott T},
    journal = {Network Neuroscience},
    number = {3},
    pages = {362--380},
    title = {{Effect of different spatial normalization approaches on tractography and structural brain networks}},
    volume = {2},
    year = {2018},
    publisher = {MIT Press}
}
""".strip()

GREVE2009 = """
@article{Greve2009,
    author = {Greve, Douglas N and Fischl, Bruce},
    journal = {Neuroimage},
    number = {1},
    pages = {63--72},
    title = {{Accurate and robust brain image alignment using boundary-based registration}},
    volume = {48},
    year = {2009},
    publisher = {Elsevier}
}
""".strip()

BRETT2001 = """
@article{Brett2001,
    author = {Brett, Matthew and Leff, Alexander P and Rorden, Chris and Ashburner, John},
    journal = {Neuroimage},
    number = {2},
    pages = {486--500},
    title = {{Spatial normalization of brain images with focal lesions using cost function masking}},
    volume = {14},
    year = {2001},
    publisher = {Academic Press}
}
""".strip()

CHUNG2019 = """
@article{Chung2019,
    author = {Chung, Jaewon and Pedigo, Benjamin D and Bridgeford, Eric W and Varjavand, Bijan K and Helm, Hayden S and Vogelstein, Joshua T},
    journal = {Journal of Machine Learning Research},
    number = {158},
    pages = {1--7},
    title = {{GraSPy: Graph Statistics in Python}},
    volume = {20},
    year = {2019}
}
""".strip()

ARROYO2019 = """
@article{Arroyo2019,
    author = {Arroyo, Jes{\'u}s and Athreya, Avanti and Cape, Joshua and Chen, Guodong and Priebe, Carey E and Vogelstein, Joshua T},
    journal = {arXiv preprint arXiv:1906.10026},
    title = {{Inference for multiple heterogeneous networks with a common invariant subspace}},
    year = {2019}
}
""".strip()

SUSSMAN2012 = """
@article{Sussman2012,
    author = {Sussman, Daniel L and Tang, Minh and Fishkind, Donniell E and Priebe, Carey E},
    journal = {Journal of the American Statistical Association},
    number = {499},
    pages = {1119--1128},
    title = {{A consistent adjacency spectral embedding for stochastic blockmodel graphs}},
    volume = {107},
    year = {2012},
    publisher = {Taylor \& Francis Group}
}
""".strip()

ROSENTHAL2018 = """
@article{Rosenthal2018,
    author = {Rosenthal, Gideon and V{\'a}{\v{s}}a, Franti{\v{s}}ek and Griffa, Alessandra and Hagmann, Patric and Amico, Enrico and Go{\~n}i, Joaqu{\'\i}n and Avidan, Galia and Sporns, Olaf},
    journal = {Nature communications},
    number = {1},
    pages = {1--12},
    volume = {9},
    title = {{Mapping higher-order relations between brain structure and function with embedded vector representations of connectomes}},
    year = {2018},
    publisher = {Nature Publishing Group}
}
""".strip()

SPORNS2004 = """
@article{Sporns2004,
    author = {Sporns, Olaf and K{\"o}tter, Rolf},
    journal = {PLoS biology},
    number = {11},
    title = {{Motifs in brain networks}},
    volume = {2},
    year = {2004},
    publisher = {Public Library of Science}
}
""".strip()

BATTISTON2017 = """
@article{Battiston2017,
    author = {Battiston, Federico and Nicosia, Vincenzo and Chavez, Mario and Latora, Vito},
    journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    number = {4},
    pages = {047404},
    title = {{Multilayer motif analysis of brain networks}},
    volume = {27},
    year = {2017},
    publisher = {AIP Publishing LLC}
}
""".strip()

AMATO2017 = """
@article{Amato2017,
    author = {Amato, Roberta and Kouvaris, Nikos E and San Miguel, Maxi and D{\'\i}az-Guilera, Albert},
    journal = {New Journal of Physics},
    number = {12},
    pages = {123019},
    title = {{Opinion competition dynamics on multiplex networks}},
    volume = {19},
    year = {2017},
    publisher = {IOP Publishing}
}
""".strip()

KOUVARIS2015 = """
@article{Kouvaris2015,
    author = {Kouvaris, Nikos E and Hata, Shigefumi and D{\'\i}az-Guilera, Albert},
    journal = {Scientific reports},
    number = {1},
    pages = {1--9},
    title = {{Pattern formation in multiplex networks}},
    volume = {5},
    year = {2015},
    publisher = {Nature Publishing Group}
}
""".strip()

SOLE2013 = """
@article{Sole2013,
    author = {Sole-Ribalta, Albert and De Domenico, Manlio and Kouvaris, Nikos E and Diaz-Guilera, Albert and Gomez, Sergio and Arenas, Alex},
    journal = {Physical Review E},
    number = {3},
    pages = {032807},
    title = {{Spectral properties of the Laplacian of multiplex networks}},
    volume = {88},
    year = {2013},
    publisher = {APS}
}
""".strip()

BULLMORE2009 = """
@article{Bullmore2009,
    author = {Bullmore, Ed and Sporns, Olaf},
    journal = {Nature reviews neuroscience},
    number = {3},
    pages = {186--198},
    title = {{Complex brain networks: graph theoretical analysis of structural and functional systems}},
    volume = {10},
    year = {2009},
    publisher = {Nature Publishing Group}
}
""".strip()

VAIANA2018 = """
@article{Vaiana2018,
    author = {Vaiana, Michael and Muldoon, Sarah Feldt},
    journal = {Journal of Nonlinear Science},
    pages = {1--23},
    title = {{Multilayer brain networks}},
    year = {2018},
    publisher = {Springer}
}
""".strip()

BRON1973 = """
@article{Bron1973,
    author = {Bron, Coen and Kerbosch, Joep},
    journal = {Communications of the ACM},
    number = {9},
    pages = {575--577},
    title = {{Algorithm 457: finding all cliques of an undirected graph}},
    volume = {16},
    year = {1973},
    publisher = {ACM New York, NY, USA}
}
""".strip()

TOMITA2006 = """
@article{Tomita2006,
    author = {Tomita, Etsuji and Tanaka, Akira and Takahashi, Haruhisa},
    journal = {Theoretical computer science},
    number = {1},
    pages = {28--42},
    title = {{The worst-case time complexity for generating all maximal cliques and computational experiments}},
    volume = {363},
    year = {2006},
    publisher = {Elsevier}
}
""".strip()

CAZALS2008 = """
@article{Cazals2008,
    author = {Cazals, Fr{\'e}d{\'e}ric and Karande, Chinmay},
    journal = {Theoretical Computer Science},
    number = {1-3},
    pages = {564--568},
    title = {{A note on the problem of reporting maximal cliques}},
    volume = {407},
    year = {2008},
    publisher = {Elsevier}
}
""".strip()

LATORA2003 = """
@article{Latora2003,
author = {Latora, Vito and Marchiori, Massimo},
journal = {The European Physical Journal B-Condensed Matter and Complex Systems},
number = {2},
pages = {249--263},
title = {{Economic small-world behavior in weighted networks}},
volume = {32},
year = {2003},
publisher = {Springer}
}
""".strip()

TELESFORD2011 = """
@article{Telesford2011,
    author = {Telesford, Qawi K and Joyce, Karen E and Hayasaka, Satoru and Burdette, Jonathan H and Laurienti, Paul J},
    journal = {Brain connectivity},
    number = {5},
    pages = {367--375},
    title = {{The ubiquity of small-world networks}},
    volume = {1},
    year = {2011},
    publisher = {Mary Ann Liebert, Inc. 140 Huguenot Street, 3rd Floor New Rochelle, NY 10801 USA}
}
""".strip()

NEWMAN2004 = """
@article{Newman2004,
    author = {Newman, Mark EJ and Girvan, Michelle},
    journal = {Physical review E},
    number = {2},
    pages = {026113},
    title = {{Finding and evaluating community structure in networks}},
    volume = {69},
    year = {2004},
    publisher = {APS}
}
""".strip()

GUIMERA2005 = """
@article{Guimera2005,
    author = {Guimera, Roger and Amaral, Luis A Nunes},
    journal = {nature},
    number = {7028},
    pages = {895--900},
    title = {{Functional cartography of complex metabolic networks}},
    volume = {433},
    year = {2005},
    publisher = {Nature Publishing Group}
}
""".strip()

DE2014 = """
@article{De2014,
    author = {de Reus, Marcel A and Saenger, Victor M and Kahn, Ren{\'e} S and van den Heuvel, Martijn P},
    journal = {Philosophical Transactions of the Royal Society B: Biological Sciences},
    number = {1653},
    pages = {20130527},
    title = {{An edge-centric perspective on the human connectome: link communities in the brain}},
    volume = {369},
    year = {2014},
    publisher = {The Royal Society}
}
""".strip()

WASSERMAN1994 = """
@article{Wasserman1994,
    author = {Wasserman, Stanley and Faust, Katherine and others},
    title = {{Social network analysis: Methods and applications}},
    volume = {8},
    year = {1994},
    publisher = {Cambridge university press}
}
""".strip()

BARRAT2004 = """
@article{Barrat2004,
    author = {Barrat, Alain and Barthelemy, Marc and Pastor-Satorras, Romualdo and Vespignani, Alessandro},
    journal = {Proceedings of the national academy of sciences},
    number = {11},
    pages = {3747--3752},
    title = {{The architecture of complex weighted networks}},
    volume = {101},
    year = {2004},
    publisher = {National Acad Sciences}
}
""".strip()

HAYASAKA2017 = """
@article{Hayasaka2017,
    author = {Hayasaka, Satoru},
    journal = {Brain connectivity},
    number = {8},
    pages = {504--514},
    title = {{Anti-Fragmentation of Resting-State Functional Magnetic Resonance Imaging Connectivity Networks with Node-Wise Thresholding}},
    volume = {7},
    year = {2017},
    publisher = {Mary Ann Liebert, Inc. 140 Huguenot Street, 3rd Floor New Rochelle, NY 10801 USA}
}
""".strip()

POWER2013 = """
@article{Power2013,
    author = {Power, Jonathan D and Schlaggar, Bradley L and Lessov-Schlaggar, Christina N and Petersen, Steven E},
    journal = {Neuron},
    number = {4},
    pages = {798--813},
    title = {{Evidence for hubs in human functional brain networks}},
    volume = {79},
    year = {2013},
    publisher = {Elsevier}
}
""".strip()

ROHE2016 = """
@article{Rohe2016,
    author = {Rohe, Karl and Qin, Tai and Yu, Bin},
    journal = {Proceedings of the National Academy of Sciences},
    number = {45},
    pages = {12679--12684},
    title = {{Co-clustering directed graphs to discover asymmetries and directional communities}},
    volume = {113},
    year = {2016},
    publisher = {National Acad Sciences}
}
""".strip()

DRAKESMITH2015 = """
@article{Drakesmith2015,
author = {Drakesmith, Mark and Caeyenberghs, Karen and Dutt, A and Lewis, G and David, AS and Jones, Derek K},
journal = {Neuroimage},
pages = {313--333},
title = {{Overcoming the effects of false positives and threshold bias in graph theoretical analyses of neuroimaging data}},
volume = {118},
year = {2015},
publisher = {Elsevier}
}
""".strip()

FORNITO2016 = """
@book{Fornito2016,
    author = {Fornito, Alex and Zalesky, Andrew and Bullmore, Edward},
    title = {{Fundamentals of brain network analysis}},
    year = {2016},
    publisher = {Academic Press}
}
""".strip()

YU2001 = """
@inproceedings{Yu2001,
    author = {Yu, Stella X and Shi, Jianbo},
    pages = {II--II},
    booktitle = {Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001},
    title = {{Understanding popout through repulsion}},
    volume = {2},
    year = {2001},
    organization = {IEEE}
}
""".strip()

STELLA2003 = """
@inproceedings{Stella2003,
    author = {Stella, X Yu and Shi, Jianbo},
    pages = {313},
    booktitle = {null},
    title = {{Multiclass spectral clustering}},
    year = {2003},
    organization = {IEEE}
}
""".strip()

ADLURU2013 = """
@inproceedings{Adluru2013,
    author = {Adluru, Nagesh and Zhang, Hui and Tromp, Do PM and Alexander, Andrew L},
    pages = {86690A},
    booktitle = {Medical Imaging 2013: Image Processing},
    title = {{Effects of DTI spatial normalization on white matter tract reconstructions}},
    volume = {8669},
    year = {2013},
    organization = {International Society for Optics and Photonics}
}
""".strip()

LEVIN2017 = """
@inproceedings{Levin2017,
    author = {Levin, Keith and Athreya, Avanti and Tang, Minh and Lyzinski, Vince and Priebe, Carey E},
    pages = {964--967},
    booktitle = {2017 IEEE International Conference on Data Mining Workshops (ICDMW)},
    title = {{A central limit theorem for an omnibus embedding of multiple random dot product graphs}},
    year = {2017},
    organization = {IEEE}
}
""".strip()

LIU2018 = """
@inproceedings{Liu2018,
    author = {Liu, Ye and He, Lifang and Cao, Bokai and Philip, S Yu and Ragin, Ann B and Leow, Alex D},
    booktitle = {Thirty-Second AAAI Conference on Artificial Intelligence},
    title = {{Multi-view multi-graph embedding for brain network clustering analysis}},
    year = {2018}
}
""".strip()

QIN2013 = """
@inproceedings{Qin2013,
    author = {Qin, Tai and Rohe, Karl},
    pages = {3120--3128},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Regularized spectral clustering under the degree-corrected stochastic blockmodel},
    year = {2013}
}
""".strip()

LASKA2017 = """
@software{Laska2017,
    author = {Jason Laska and Manjari Narayan},
    title = {{skggm 0.2.7: A scikit-learn compatible package for Gaussian and related Graphical Models}},
    month = jul,
    year = 2017,
    publisher = {Zenodo},
    doi = {10.5281/zenodo.830033},
    url = {https://doi.org/10.5281/zenodo.830033}
}
""".strip()
