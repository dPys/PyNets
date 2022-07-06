"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pickle5 as pickle
import sqlalchemy as sa
import json
from sqlalchemy import Column, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import func, select
from sqlalchemy.types import BLOB, JSON, DateTime, Float, Integer, String

from pynets.core.utils import load_runconfig

pickle.HIGHEST_PROTOCOL = 5
warnings.filterwarnings("ignore")

base = declarative_base()


@dataclass
class ConnectomeEnsemble(base):
    """
    A data class to manage the metadata of a connectome ensemble from a single subject.

    Parameters
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
    parcellation_meta: str
        A value for the parcellation hyperparameter.
    subnet_meta: str
        A value for the subnetwork hyperparameter.
    template: str
        A value for the template hyperparameter.
    thr_type: str
        A value for the threshold type hyperparameter.
    thr: float
        A value for the threshold hyperparameter.
    node_type: str
        A value for the node type hyperparameter.
    signal_meta: str
        A value for the signal hyperparameter.
    traversal_meta: str
        A value for the traversal hyperparameter.
    minlength_meta: int
        A value for the minlength hyperparameter.
    hpass: float
        A value for the float hyperparameter.
    model_meta: str
        A value for the model hyperparameter.
    granularity_meta: str
        A value for the granularity hyperparameter.
    smooth_meta: str
        A value for the smooth hyperparameter.
    error_margin_meta: float
        A value for the error margin hyperparameter.
    """

    __tablename__ = "connectome_ensemble"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement="auto")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), nullable=False)
    subject_id = Column(String(255), nullable=False)
    session = Column(String(255), nullable=False)
    modality = Column(String(255), nullable=False)
    embed_meta = Column(String(255), nullable=False)
    parcellation_meta = Column(String(255), nullable=False)
    subnet_meta = Column(String(255), nullable=False)
    template = Column(String(255), nullable=False, server_default="MNI152_T1")
    thr_type = Column(String(255), nullable=True, server_default="NULL")
    thr = Column(Float, nullable=True, server_default="NULL")
    node_type = Column(String(255), nullable=False, server_default="parcels")
    data = Column(BLOB, nullable=True)
    signal_meta = Column(String(255), nullable=True)
    traversal_meta = Column(String(255), nullable=True)
    minlength_meta = Column(Integer, nullable=True)
    hpass_meta = Column(Integer, nullable=True)
    model_meta = Column(String(255), nullable=False)
    granularity_meta = Column(Integer, nullable=False)
    smooth_meta = Column(Float, nullable=True)
    error_margin_meta = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"ConnectomeEnsemble\nSubject:{self.subject_id}\n"
            f"Table Creation:{self.created_at}\n"
            f"Table Update:{self.updated_at}\n"
            f"Modality:{self.modality}\n"
            f"Embedding:{self.embed_meta}\n"
            f"Parcellation:{self.parcellation_meta}\n"
            f"Network:{self.subnet_meta}\n"
            f"Template:{self.template}\n"
            f"Threshold Type:{self.thr_type}\n"
            f"Threshold:{self.thr}\n"
            f"Node Type:{self.node_type}\n"
            f"Signal:{self.signal_meta}\n"
            f"Traversal:{self.traversal_meta}\n"
            f"Minlength:{self.minlength_meta}\n"
            f"High-Pass:{self.hpass_meta}\n"
            f"Model:{self.model_meta}\n"
            f"Granularity:{self.granularity_meta}\n"
            f"Smooth:{self.smooth_meta}\n"
            f"Error Margin:{self.error_margin_meta}\n"
        )


def gen_session(output_dir: str) -> Session:
    hardcoded_params = load_runconfig()
    RDBMS = hardcoded_params["sql_config"]["RDBMS"]
    DATABASE_URI = f"{RDBMS}:////{output_dir}/pynets.db"
    engine = create_engine(DATABASE_URI)
    base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    session = session()
    return session


def insert_dwi_func(pkl_file_path: str, session: Session) -> dict:
    with open(pkl_file_path, "rb") as f:
        sub_dict_clean = pickle.load(f)
    f.close()

    elements = []
    nodes = {}
    return iter_dict(sub_dict_clean, elements, nodes, session)


def fetch_dwi_func(
    model: str,
    parcellation: str,
    subnet: str,
    embedding: str,
    output_dir: str,
    thr_type: str,
    thr: float,
    template: str,
) -> dict:
    ce = ConnectomeEnsemble()
    subject_id_session_modality_embedding_set = set()
    subject_id_session_modality_embedding_dict = {}
    session = gen_session(output_dir)
    result = session.query(ce).filter_by(
        modality=model,
        parcellation_meta=parcellation,
        embed_meta=embedding,
        template=template,
        thr_type=thr_type,
        thr=thr,
    )
    for row in result:
        subject_id_session_modality_embedding_set.add(
            (row.subject_id, row.session, row.modality, row.embed_meta)
        )

    for item in subject_id_session_modality_embedding_set:
        subject_id_session_modality_embedding_dict[item] = []

    for row in result:
        index_data_dict = {}
        tuple_dict = {}
        tuple_connectome = ()
        if row.modality == "dwi":
            tuple_connectome = (
                str(row.signal_meta),
                str(row.minlength_meta),
                str(row.model_meta),
                str(row.granularity_meta),
                str(row.subnet_meta),
                str(row.tolerance_meta),
            )
        elif row.modality == "func":
            tuple_connectome = (
                str(row.signal_meta),
                str(row.minlength_meta),
                str(row.model_meta),
                str(row.granularity_meta),
                str(row.subnet_meta),
                str(row.tolerance_meta),
            )
        if row.data_file_path == "NaN":
            tuple_dict[tuple_connectome] = "NaN"
        else:
            index_data_dict["index"] = str(row.node_ref["index"])
            byte_data_array = row.data
            converted_data_array = np.array(
                np.frombuffer(byte_data_array), ndmin=2
            ).T
            index_data_dict["data"] = converted_data_array
            tuple_dict[tuple_connectome] = index_data_dict
        subject_id_session_modality_embedding_tuple = (
            row.subject_id,
            row.session,
            row.modality,
            row.embed_meta,
        )
        subject_id_session_modality_embedding_dict[
            subject_id_session_modality_embedding_tuple
        ].append(tuple_dict)

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


# class EnsembleReaderWriter(object):

#     def __init__(self):
#         self.ce =

#     def to_dict(self):
#         return {
#             "subject_id": self.ce.subject_id,
#             "session": self.ce.session,
#             "modality": self.ce.modality,
#             "embed": self.ce.embed_meta,
#             "parcellation": self.ce.parcellation_meta,
#             "subnet": self.ce.subnet_meta,
#             "template": self.ce.template,
#             "thr_type": self.ce.thr_type,
#             "thr": self.ce.thr,
#             "node_type": self.ce.node_type,
#             "signal": self.ce.signal_meta,
#             "traversal": self.ce.traversal_meta,
#             "minlength": self.ce.minlength_meta,
#             "hpass": self.ce.hpass_meta,
#             "model": self.model_meta,
#             "granularity": self.granularity_meta,
#             "smooth": self.smooth_meta,
#             "error_margin": self.error_margin_meta,
#         }

#     def from_dict(self, d):
#         self.subject_id = d["subject_id"]
#         self.session = d["session"]
#         self.modality = d["modality"]
#         self.embed_meta = d["embed"]
#         self.parcellation_meta = d["parcellation"]
#         self.subnet_meta = d["subnet"]
#         self.template = d["template"]
#         self.thr_type = d["thr_type"]
#         self.thr = d["thr"]
#         self.node_type = d["node_type"]
#         self.signal_meta = d["signal"]
#         self.traversal_meta = d["traversal"]
#         self.minlength_meta = d["minlength"]
#         self.hpass_meta = d["hpass"]
#         self.model_meta = d["model"]
#         self.granularity_meta = d["granularity"]
#         self.smooth_meta = d["smooth"]
#         self.error_margin_meta = d["error_margin"]


#     def to_json(self):
#         return json.dumps(self.to_dict())

#     def from_json(self, json_str):
#         return json.loads(json_str)


def iter_dict(d, elements, nodes, session):
    for k, v in d.items():
        if isinstance(v, dict):
            elements.append(k)
            iter_dict(v, elements, nodes, session)
            del elements[-1]
        else:
            if k == "index" or k == "labels" or k == "coords":
                nodes[k] = v
            elif k == "data":
                data_array = np.load(v)
                new_entry = ConnectomeEnsemble()
                new_entry.subject_id = elements[0]
                new_entry.session = elements[1]
                new_entry.modality = elements[2]
                new_entry.embed_meta = elements[3]
                if elements[2] == "func":
                    new_entry.signal_meta = (elements[4])[0]
                    new_entry.hpass_meta = (elements[4])[1]
                elif elements[2] == "dwi":
                    new_entry.traversal_meta = (elements[4])[0]
                    new_entry.minlength_meta = (elements[4])[1]
                new_entry.model_meta = (elements[4])[2]
                new_entry.granularity_meta = (elements[4])[3]
                new_entry.subnet_meta = (elements[4])[4]
                new_entry.tolerance_meta = (elements[4])[5]
                new_entry.node_ref = nodes
                new_entry.data = bytearray(data_array)
                new_entry.data_file_path = elements[5]
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
                if elements[2] == "func":
                    new_entry.signal_meta = (elements[4])[0]
                    new_entry.hpass_meta = (elements[4])[1]
                elif elements[2] == "dwi":
                    new_entry.traversal_meta = (elements[4])[0]
                    new_entry.minlength_meta = (elements[4])[1]
                new_entry.model_meta = (elements[4])[2]
                new_entry.granularity_meta = (elements[4])[3]
                new_entry.subnet_meta = (elements[4])[4]
                new_entry.tolerance_meta = (elements[4])[5]
                new_entry.data_file_path = elements[5]
                session.add(new_entry)
                session.commit()
                del elements[-1], elements[-1]
