import pickle
import config as C

from Bio import SeqIO
from keras.models import load_model


def read_fasta_file(filename):
    '''
    Returns list of sequence records in a fasta file
    '''
    seqs = []
    for sr in SeqIO.parse(filename, "fasta"):
        srecord = sr.seq
        seqs.append(srecord)
    return seqs


def serialize_descriptor_vector(dvec, o_file=None):
    if o_file is None:
        path = C.serde_model_path + C.model_name + ".pickle"
    else:
        path = C.serde_model_path + o_file + ".pickle"
    output = open(path, 'wb')
    pickle.dump(dvec, output)
    output.close()


def deserialize_descriptor_vector(model_name):
    path = C.serde_model_path + model_name + ".pickle"
    input = open(path, 'rb')
    dvec = pickle.load(input)
    return dvec


def serialize_model(model, o_file=None):
    if o_file is None:
        model.save(C.model_path)
    else:
        model.save(o_file)


def deserialize_model(o_file=None):
    if o_file is None:
        model = load_model(C.model_path)
    else:
        model = load_model(o_file)
    return model


if __name__ == "__main__":
    filename = "ls_orchid.fasta"
    seqs = read_fasta_file(filename)
    for s in seqs:
        print(str(s))
