import pkgutil
import io
from pathlib import Path
import pandas as pd
import pynets
from pynets import nodemaker
from nilearn import datasets

def test_coord_maker():
    NETWORK='DMN'
    mask = Path(__file__).parent/"examples"/"997"/"pDMN_3_bin.nii.gz"
    atlas_select = 'coords_power_2011'
    [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)

    if atlas_name == 'Power 2011 atlas':
        network_coords_ref = NETWORK + '_coords.csv'
        atlas_coords = pkgutil.get_data("pynets", "rsnrefs/" + network_coords_ref)
        df = pd.read_csv(io.BytesIO(atlas_coords)).ix[:,0:4]
        i=1
        net_coords = []
        ix_labels = []
        for i in range(len(df)):
            x = int(df.ix[i,1])
            y = int(df.ix[i,2])
            z = int(df.ix[i,3])
            net_coords.append((x, y, z))
            ix_labels.append(i)
            i = i + 1
            label_names=ix_labels

    if label_names!=ix_labels:
        try:
            label_names=label_names.tolist()
        except:
            pass
        label_names=[label_names[i] for i in ix_labels]

    [net_coords, label_names] = nodemaker.coord_masker(str(mask), net_coords, label_names)
    assert net_coords is not None
    assert label_names is not None
