import warnings
import numpy as np
import pickle5 as pickle
from sqlalchemy import create_engine, Column
from sqlalchemy.types import Integer, Float, Integer, JSON, BLOB, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select, func
from sqlalchemy.ext.declarative import declarative_base

pickle.HIGHEST_PROTOCOL = 5
warnings.filterwarnings("ignore")

base = declarative_base()


class ConnectomeEnsemble(base):
    """
    A data class to manage the metadata of a connectome ensemble from a single subject.

    Attributes
    ----------
    id : int
        The id of the connectome sample.
    created_at : datetime
        Datetime of table creation.
    updated_at : datetime
        Datetime of last table update.
    subject_id : int
        The id of the subject.
    session: int
        The id of the session.
    modality: str
        The modality of the connectome sample.
    embed_meta: str
        A value for the embedding type hyperparameter.
    net_meta: str
        A value for the network type hyperparameter.
    template: str
        A value for the template hyperparameter.
    thr_type: str
        A value for the threshold type hyperparameter.
    thr: float
        A value for the threshold hyperparameter.
    node_type: str
        A value for the node type hyperparameter.
    data_file_path: str
        The path to the data file.
    signal_meta: str
        A value for the signal hyperparameter.
    minlength_meta: str
        A value for the minlength hyperparameter.
    model_meta: str
        A value for the model hyperparameter.
    granularity_meta: str
        A value for the granularity hyperparameter.
    tolerance_meta: str
        A value for the tolerance hyperparameter.
    """
    __tablename__ = 'connectome_ensemble'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement="auto")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), nullable=False)
    subject_id = Column(String(255), nullable=False)
    session = Column(String(255), nullable=False)
    modality = Column(String(255), nullable=False)
    embed_meta = Column(String(255), nullable=False)
    net_meta = Column(String(255), nullable=False)
    template = Column(String(255), nullable=False)
    thr_type = Column(String(255), nullable=False)
    thr = Column(Float, nullable=True)
    node_type = Column(String(255), nullable=False)
    data = Column(BLOB, nullable=True)
    signal_meta = Column(String(255), nullable=True)
    minlength_meta = Column(Integer, nullable=True)
    model_meta = Column(String(255), nullable=False)
    granularity_meta = Column(Integer, nullable=False)
    tolerance_meta = Column(Float, nullable=True)

    def __repr__(self):
        return f"ConnectomeEnsemble\nSubject:{self.subject_id}\nTable Creation:{self.created_at}\nTable Update:{self.updated_at}\nModality:{self.modality}\nEmbedding:{self.embed_meta}\nNetwork:{self.net_meta}\nTemplate:{self.template}\nThreshold Type:{self.thr_type}\nThreshold:{self.thr}\nNode Type:{self.node_type}\nSignal:{self.signal_meta}\nMinlength:{self.minlength_meta}\nModel:{self.model_meta}\nGranularity:{self.granularity_meta}\nTolerance:{self.tolerance_meta}"


def gen_session(output_dir: str):
    DATABASE_URI = f"sqlite:////{output_dir}/pynets.db"
    engine = create_engine(DATABASE_URI)
    base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    session = session()
    return session


def insert_dwi_func(pkl_file_path: str, session):
    with open(pkl_file_path, "rb") as f:
        sub_dict_clean = pickle.load(f)
    f.close()

    elements = []
    nodes = {}
    return iter_dict(sub_dict_clean, elements, nodes, session)


def fetch_dwi_func(mod: str, net: str, embedding: str, db_path: str, thr_type: str):
    ce = ConnectomeEnsemble()
    subject_id_session_modality_embedding_set = set()
    subject_id_session_modality_embedding_dict = {}
    session = gen_session(db_path)
    template = 'MNI152_T1'
    thr_type = 'MST'
    result = session.query(ce).filter_by(modality=mod, net_meta=net,
                                         embed_meta=embedding,
                                         template=template, thr_type=thr_type)
    for row in result:
        subject_id_session_modality_embedding_set.add((row.subject_id,
                                                       row.session,
                                                       row.modality,
                                                       row.embed_meta))

    for item in subject_id_session_modality_embedding_set:
        subject_id_session_modality_embedding_dict[item] = []

    for row in result:
        index_data_dict = {}
        tuple_dict = {}
        tuple_connectome = ()
        if row.modality == 'dwi':
            tuple_connectome = (str(row.signal_meta), str(row.minlength_meta),
                                str(row.model_meta),
                                str(row.granularity_meta), str(row.net_meta),
                                str(row.tolerance_meta))
        elif row.modality == 'func':
            tuple_connectome = (str(row.signal_meta), str(row.minlength_meta),
                                str(row.model_meta),
                                str(row.granularity_meta), str(row.net_meta),
                                str(row.tolerance_meta))
        if row.data_file_path == 'NaN':
            tuple_dict[tuple_connectome] = 'NaN'
        else:
            index_data_dict['index'] = str(row.node_ref["index"])
            byte_data_array = row.data
            converted_data_array = np.array(np.frombuffer(byte_data_array),
                                            ndmin=2).T
            index_data_dict['data'] = converted_data_array
            tuple_dict[tuple_connectome] = index_data_dict
        subject_id_session_modality_embedding_tuple = (row.subject_id,
                                                       row.session,
                                                       row.modality,
                                                       row.embed_meta)
        subject_id_session_modality_embedding_dict[
            subject_id_session_modality_embedding_tuple].append(tuple_dict)

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


def iter_dict(d, elements, nodes, session):
    for k, v in d.items():
        if isinstance(v, dict):
            elements.append(k)
            iter_dict(v, elements, nodes, session)
            del elements[-1]
        else:
            if k == 'index' or k == "labels" or k == "coords":
                nodes[k] = v
            elif k == 'data':
                data_array = np.load(v)
                new_entry = ConnectomeEnsemble()
                new_entry.subject_id = elements[0]
                new_entry.session = elements[1]
                new_entry.modality = elements[2]
                new_entry.embed_meta = elements[3]
                if elements[2] == 'func':
                    new_entry.signal_meta = (elements[4])[0]
                    new_entry.hpass_meta = (elements[4])[1]
                elif elements[2] == 'dwi':
                    new_entry.traversal_meta = (elements[4])[0]
                    new_entry.minlength_meta = (elements[4])[1]
                new_entry.model_meta = (elements[4])[2]
                new_entry.granularity_meta = (elements[4])[3]
                new_entry.net_meta = (elements[4])[4]
                new_entry.tolerance_meta = (elements[4])[5]
                new_entry.node_ref = nodes
                new_entry.data = bytearray(data_array)
                new_entry.data_file_path = elements[5]
                new_entry.template = 'MNI152_T1'
                new_entry.thr_type = 'MST'
                new_entry.thr = 1.0
                new_entry.node_type = 'parcels'
                session.add(new_entry)
                session.commit()
            else:
                elements.append(k)
                elements.append(v)
                new_entry = ConnectomeEnsemble()
                new_entry.subject_id = elements[0]
                new_entry.session = elements[1]
                new_entry.modality = elements[2]
                new_entry.embed_meta = elements[3]
                if elements[2] == 'func':
                    new_entry.signal_meta = (elements[4])[0]
                    new_entry.hpass_meta = (elements[4])[1]
                elif elements[2] == 'dwi':
                    new_entry.traversal_meta = (elements[4])[0]
                    new_entry.minlength_meta = (elements[4])[1]
                new_entry.model_meta = (elements[4])[2]
                new_entry.granularity_meta = (elements[4])[3]
                new_entry.net_meta = (elements[4])[4]
                new_entry.tolerance_meta = (elements[4])[5]
                new_entry.data_file_path = elements[5]
                new_entry.template = 'MNI152_T1'
                new_entry.thr_type = 'MST'
                new_entry.thr = 1.0
                new_entry.node_type = 'parcels'
                session.add(new_entry)
                session.commit()
                del elements[-1], elements[-1]
