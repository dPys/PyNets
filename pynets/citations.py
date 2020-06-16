from pynets.__about__ import __version__
from datetime import datetime

PYNETS = """
@CONFERENCE{{dPys,
    title = {{PyNets v{version}: A Reproducible Workflow for Structural and 
    Functional Connectome Ensemble Learning'}},
    author = {{Pisner, D.}},
    publisher = {{Poster session presented at: Annual Meeting of the Organization 
    for Human Brain Mapping}},
    url = {{https://github.com/dPys/PyNets}},
    note = {{[Online; accessed {datestamp}]}},
    year = {{2020}},
    month = {{June}}
}}
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
