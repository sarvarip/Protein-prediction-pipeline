from pydpi.protein import AAComposition
#from pydpi.drug import topology
#from pydpi.pydrug import Chem


def extract_named_descriptors_of_seq(sequence):
    '''
    Returns a map ("descriptor" -> value) of descriptors when given a sequence of aminoacids (string)
    :param sequence:
    :return:
    '''
    #mol = Chem.MolFromSequence(str(sequence))
    res = {}
    sequence=str(sequence)
    print(sequence)
    res = AAComposition.CalculateAAComposition(sequence)
    #res.update(topology.GetTopology(mol))
    return res


def num_vector_from_descriptor_vector(descriptor_vector):
    '''
    Given a sequence (map) of ("descriptor_name" -> value), returns a vector of values
    :param descriptor_vector:
    :return:
    '''
    x = []
    for k, v in descriptor_vector.items():
        x.append(v)
    return x

if __name__ == "__main__":
    print("TODO")
