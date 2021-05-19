from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.sql import select
import numpy as np
import pynets.core.interfaces
from pynets.core.interfaces import ConnectomeEnsemble as ce
import dill

def connection():
    DATABASE_URI = 'sqlite:////Users/Keval/Documents/College Senior Classes/Research/pynets_db_work/pynets.db'
    engine = create_engine(DATABASE_URI)
    pynets.core.interfaces.base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    session = session()
    return session

def insert_dwi_func(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
      sub_dict_clean = dill.load(f)
    f.close()

    elements = []
    nodes = {}
    iter_dict(sub_dict_clean, elements, nodes)

def fetch_dwi_func(mod, net, embedding, temp, thrType):
    subject_id_session_modality_embedding_set = set()
    subject_id_session_modality_embedding_dict = {}
    session = connection()
    temp = 'MNI152_T1'
    thrType = 'MST'
    result = session.query(ce).filter_by(modality = mod, net_meta = net, embed_meta = embedding, template = temp, thr_type = thrType)
    for row in result:
        subject_id_session_modality_embedding_set.add((row.subject_id, row.session, row.modality, row.embed_meta))

    for item in subject_id_session_modality_embedding_set:
        subject_id_session_modality_embedding_dict[item] = []

    for row in result:
        index_data_dict = {}
        tuple_dict = {}
        tuple_connectome = ()
        if row.modality == 'dwi':
            tuple_connectome = (str(row.extract_meta), str(row.minlength_meta), str(row.model_meta),
            str(row.granularity_meta), str(row.net_meta), str(row.tolerance_meta))
        elif row.modality == 'func':
            tuple_connectome = (str(row.extract_meta), str(row.minlength_meta), str(row.model_meta),
            str(row.granularity_meta), str(row.net_meta), str(row.tolerance_meta))
        if row.data_file_path == 'NaN':
            tuple_dict[tuple_connectome] = 'NaN'
        else:
            index_data_dict['index'] = str(row.node_ref["index"])
            byte_data_array = row.data
            converted_data_array = np.array(np.frombuffer(byte_data_array), ndmin=2).T
            index_data_dict['data'] = converted_data_array
            tuple_dict[tuple_connectome] = index_data_dict
        subject_id_session_modality_embedding_tuple = (row.subject_id, row.session, row.modality, row.embed_meta)
        subject_id_session_modality_embedding_dict[subject_id_session_modality_embedding_tuple].append(tuple_dict)

    combined_tuple_dict = {}
    refined_tuple_dict = {}
    for key in subject_id_session_modality_embedding_dict:
        for dictionary in subject_id_session_modality_embedding_dict[key]:
            combined_tuple_dict.update(dictionary)
        refined_tuple_dict[key] = combined_tuple_dict
        combined_tuple_dict = {}


    subject_id_session_dict = {}
    for key in refined_tuple_dict:
        embedding_dict = {}
        modality_dict = {}
        embedding_dict[key[-1]] = refined_tuple_dict[key]
        modality_dict[key[-2]] = embedding_dict
        subject_id_session_dict[(key[0], key[1])] = modality_dict

    final_dict = {}
    for key in subject_id_session_dict:
        session_dict = {}
        subject_id_dict = {}
        session_dict[key[-1]] = subject_id_session_dict[key]
        subject_id_dict[key[0]] = session_dict
        final_dict.update(subject_id_dict)

    return final_dict

def iter_dict(d, elements, nodes):
 session = connection()
 for k, v in d.items():
    if isinstance(v, dict):
      elements.append(k)
      iter_dict(v, elements, nodes)
      del elements[-1]
    else:
      if k == 'index' or k == "labels" or k == "coords":
        nodes[k] = v
      elif k == 'data':
        data_array = np.load(v)
        new_entry = ce()
        new_entry.subject_id = elements[0]
        new_entry.session = elements[1]
        new_entry.modality = elements[2]
        new_entry.embed_meta = elements[3]
        new_entry.extract_meta = (elements[4])[0]
        if elements[2] == 'func':
            new_entry.hpass_meta = (elements[4])[1]
        elif elements[2] == 'dwi':
            new_entry.minlength_meta = (elements[4])[1]
        new_entry.model_meta = (elements[4])[2]
        new_entry.granularity_meta = (elements[4])[3]
        new_entry.net_meta = (elements[4])[4]
        new_entry.tolerance_meta = (elements[4])[5]
        new_entry.node_ref = nodes
        new_entry.data = bytearray(data_array)
        new_entry.data_file_path = v
        new_entry.template = 'MNI152_T1'
        new_entry.thr_type = 'MST'
        new_entry.thr = '1.0'
        new_entry.node_type = 'parcels'
        session.add(new_entry)
        session.commit()
      else:
        elements.append(k)
        elements.append(v)
        new_entry = ce()
        new_entry.subject_id = elements[0]
        new_entry.session = elements[1]
        new_entry.modality = elements[2]
        new_entry.embed_meta = elements[3]
        new_entry.extract_meta = (elements[4])[0]
        if elements[2] == 'func':
            new_entry.hpass_meta = (elements[4])[1]
        elif elements[2] == 'dwi':
            new_entry.minlength_meta = (elements[4])[1]
        new_entry.model_meta = (elements[4])[2]
        new_entry.granularity_meta = (elements[4])[3]
        new_entry.net_meta = (elements[4])[4]
        new_entry.tolerance_meta = (elements[4])[5]
        new_entry.data_file_path = elements[5]
        new_entry.template = 'MNI152_T1'
        new_entry.thr_type = 'MST'
        new_entry.thr = '1.0'
        new_entry.node_type = 'parcels'
        session.add(new_entry)
        session.commit()
        del elements[-1], elements[-1]

