from __future__ import division
import time
import math
import numpy as np
from random import shuffle
from keras.models import model_from_json
import pickle
import random
from keras.models import model_from_json
from keras.optimizers import Adam
from multiprocessing import Value, Pool, Lock
from multiprocessing import Manager
from itertools import compress

import numpy as np

import inputoutput as IO
import featurex_pipeline_v3_pseaac_aac as FX #CHOOSE A FEATURE EXTRACTOR
import modelbuilder_pipeline as trainer

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

minlength = 3 #do not go below 3, otherwise there will be error because lambda is set to 3, and also in featurex_pipeline_v3 there is an if clause
seed = 7

'''User interface'''

'''Options start'''

vector_name = "pipeline_default"
model_name = "pipeline_default"
numcores = 4

#kfolds(0) or test set already provided(1) or Crossval(2) or None(3) or Merge provided test set with train set and do kfolds(4) or Crossval(5); or Predict(6) NOTE that class imbalance occurs/may occur during options 1! Choice zero maintains class balance as "The folds are made by preserving the percentage of samples for each class."
#By the same way, now choice 2 and 3 also maintains stratification (class balance)!
choice = 0
reduce_by_similarity = 1
use_random_small_sequence_negative = 0
already_extracted = 0

'''No need to fill next two lines if choice 6 is chosen'''

pos_input_name = "insert_name"
neg_input_name = "insert_name"

'''Only If choice 2/4/5 is chosen'''

already_extracted_test = 0
pos_input_test = "insert_name"
neg_input_test = "insert_name"

'''Only If prediction (choice 6) is chosen'''

predict_pos_input_name = "insert_name"
predict_neg_input_name = "insert_name"
known_classes = 1

'''Only change if you get error regarding the sampling or if you are notified that your dataset is not balanced and asked to increase negative dataset sampling'''

sc_1 = 2
sc_2 = 1.25

'''Options end'''

'''User interface'''

old_dataset = 0 #is zero by default. This needs to be set to 1 if the vectors were extracted in the old way and some are of zero size (happens when the sequence has invalid characters)

std_name = "std_" + model_name + "_model"
mean_name = "mean_" + model_name + "_model"

if choice!=6:
    pos_name = "pos_" + vector_name
    neg_name = "neg_" + vector_name

if choice==1 or choice==4 or choice==5:
    postest = pos_name + "_test"
    negtest = neg_name + "_test"

if choice==6:
    reduce_by_similarity = 0
    use_random_small_sequence_negative = 0
    predict_pos_name = "pos_" + vector_name + "_predict"
    predict_neg_name = "neg_" + vector_name + "_predict"

posfile = pos_input_name
negfile = neg_input_name

def main(model=None):

    if choice==6:
        if already_extracted==0:
            extract_descriptors_from_file_to_pickle(predict_pos_input_name, predict_pos_name)
            extract_descriptors_from_file_to_pickle(predict_neg_input_name, predict_neg_name)
        pos_dvec = IO.deserialize_descriptor_vector(predict_pos_name)
        neg_dvec = IO.deserialize_descriptor_vector(predict_neg_name)

    if choice!=6:
        if already_extracted==0:
            pos_samples = extract_descriptors_from_file_to_pickle(pos_input_name, pos_name)
            if use_random_small_sequence_negative==0:
                extract_descriptors_from_file_to_pickle(neg_input_name, neg_name, pos_samples)
        if choice==1 or choice==4 or choice==5:
            if already_extracted_test == 0:
                extract_descriptors_from_file_to_pickle(pos_input_test, postest)
                extract_descriptors_from_file_to_pickle(neg_input_test, negtest)


        print("Deserializing descriptor vectors...")
        pos_dvec = IO.deserialize_descriptor_vector(pos_name)

        if use_random_small_sequence_negative!=0:
            neg_dvec = IO.deserialize_descriptor_vector("neg_pipeline_complete_anticancer") #same as neg_cytotoxic
            if len(neg_dvec) >= len(pos_dvec):
                neg_dvec = neg_dvec[:len(pos_dvec)]
            else:
                print("Set use_random_small_sequence_negative to zero, because that pickle file does not contain enough samples to maintain alanced classes! Use CTRL-C to quit!")
                input()
        else:
            neg_dvec = IO.deserialize_descriptor_vector(neg_name)
            if len(neg_dvec)!=len(pos_dvec):
                print("Warning! Class balance is no achieved! Increase negative dataset sampling! Use CTRL-C to quit!")
                print("Negative dataset length: %d" %(len(neg_dvec)))
                print("Positive dataset length: %d" %(len(pos_dvec)))
                input()

        if choice==1 or choice==4 or choice==5:
            pos_dvec_test = IO.deserialize_descriptor_vector(postest)
            neg_dvec_test = IO.deserialize_descriptor_vector(negtest)

        print("Deserializing descriptor vectors...OK")
        print("")

    print("Extracting numerical vectors...")
    # maybe save these too separately

    #'''Choosing to train only with certain features'''

    #mask = {'T', 'P', 'G', 'D', 'Q', 'C', 'E', 'M', 'K'}
    #pos_dvec = [{key: dvec[key] for key in dvec.keys() & mask} for dvec in pos_dvec]
    #neg_dvec = [{key: dvec[key] for key in dvec.keys() & mask} for dvec in neg_dvec]

    pos_nmat = []
    for dvec in pos_dvec:
        if dvec is None:
            continue
        pos_nvec = FX.num_vector_from_descriptor_vector(dvec)
        pos_nmat.append(pos_nvec)

    neg_nmat = []
    for dvec in neg_dvec:
        if dvec is None:
            continue
        neg_nvec = FX.num_vector_from_descriptor_vector(dvec)
        neg_nmat.append(neg_nvec)

    if choice==1 or choice==4 or choice==5:
        pos_nmat_test = []
        for dvec in pos_dvec_test:
            if dvec is None:
                continue
            pos_nvec_test = FX.num_vector_from_descriptor_vector(dvec)
            pos_nmat_test.append(pos_nvec_test)

        neg_nmat_test = []
        for dvec in neg_dvec_test:
            if dvec is None:
                continue
            neg_nvec_test = FX.num_vector_from_descriptor_vector(dvec)
            neg_nmat_test.append(neg_nvec_test)

    print("Extracting numerical vectors...OK")
    print("")

    print("Preparing training and label data...")

    # Prepare labels
    pos_y_batch = [1 for _ in pos_nmat]
    neg_y_batch = [0 for _ in neg_nmat]

    if choice==1 or choice==4 or choice==5:
        pos_y_batch_test = [1 for _ in pos_nmat_test]
        neg_y_batch_test = [0 for _ in neg_nmat_test]

    # Append training data and labels, shuffle is done is kfolds
    neg_nmat.extend(pos_nmat)
    x = neg_nmat
    neg_y_batch.extend(pos_y_batch)
    y = neg_y_batch

    if choice==1 or choice==4 or choice==5:
        neg_nmat_test.extend(pos_nmat_test)
        x_test = neg_nmat_test
        neg_y_batch_test.extend(pos_y_batch_test)
        y_test = neg_y_batch_test

        if choice == 4 or choice==5:
            x.extend(x_test)
            y.extend(y_test)
            if choice==4:
                trained_M, mean, std = do_kfolds(x,y)
            if choice==5:
                trained_M, mean, std = do_crossval(x,y)

    print("Preparing training and label data...OK")
    print("")

    if choice==6:

        #json_file = open('./models/' + model_name + '.json', 'r')
        json_file = open('./' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        #loaded_model.load_weights("./models/" + model_name + ".h5")
        loaded_model.load_weights("./" + model_name + ".h5")
        print("Loaded model from disk")

        optim = Adam(lr=0.01, beta_1=0.95)

        loaded_model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

        #path = path = "C:/Users/Peter/Desktop/MIT/code/models/" + std_name + ".pickle"
        path = path = "./" + std_name + ".pickle"
        array_file = open(path, 'rb')
        std = pickle.load(array_file)

        #path = path = "C:/Users/Peter/Desktop/MIT/code/models/" + mean_name + ".pickle"
        path = path = "./" + mean_name + ".pickle"
        array_file = open(path, 'rb')
        mean = pickle.load(array_file)

        x -= mean
        x /= std

        result = loaded_model.predict(x)
        print("Probabilities:")
        print(result)
        classes = loaded_model.predict_classes(x)
        classes = np.array(classes)
        classes = classes.ravel()
        print("Calsses:")
        print(classes)

        if known_classes == 1:

            get_performance_vals(y, classes)

    # 10-folds cross-validation
    if choice==0:

        if old_dataset == 1:

            no_features = 114 #MODIFY THIS ACCORDINGLY

            x=np.array([np.array(xi).T for xi in x])
            remain = x.shape[0]
            num = []
            for i in range(remain):
                if x[i].shape[0] != no_features:
                    num.extend([i])
            print(len(num))
            x = np.delete(x, num, 0)
            y = np.delete(y, num, 0)
            x=np.array([np.array(xi).T for xi in x]) #needed to be done again for some reason
            print(x.size)
            remain2 = x.shape[0]
            x.reshape(remain2,no_features)

        trained_M, mean, std = do_kfolds(x,y)

    if choice==1:

        x = np.array(x)
        y = np.array(y)
        x_train = x
        y_train = y
        mean = np.mean(x_train, axis = 0)
        std = np.std(x_train, axis = 0)
        x_train -= mean
        eps = 10**-5
        std = std + eps
        x_train /= std

        x_test -= mean
        x_test /= std

        print("Training model on data...")
        s_training = time.time()
        M = trainer.build_sequential_model(rate = 0.3, shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)
        e_training = time.time()
        print("Training model on data...OK, took: " + str((e_training - s_training)))

        print("Classifying data...")
        s_classify = time.time()
        #scores = trained_M.predict_with_model(x_test)
        classes = trained_M.predict_classes(x_test)
        classes = np.array(classes)
        classes = classes.ravel()
        e_classify = time.time()
        print("Classifying data...OK, took: " + str((e_classify - s_classify)))

        mcc, accuracy, fscore, precision, recall = get_performance_vals(y_test, classes)


    if choice==2:

        trained_M, mean, std = do_crossval(x,y)

    if choice==3:

        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=seed,stratify=y)
        mean = np.mean(x_train, axis = 0)
        std = np.std(x_train, axis = 0)
        x_train -= mean
        eps = 10**-5
        std = std + eps
        x_train /= std

        x_test -= mean
        x_test /= std

        print("Training model on data...")
        s_training = time.time()
        M = trainer.build_sequential_model(rate = 0.3, shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)

        print("Classifying data...")
        s_classify = time.time()
        #scores = trained_M.(x_test)
        classes = trained_M.predict_classes(x_test)
        classes = np.array(classes)
        classes = classes.ravel()

        e_classify = time.time()
        print("Classifying data...OK, took: " + str((e_classify - s_classify)))

        mcc, accuracy, fscore, precision, recall = get_performance_vals(y_test, classes)

    if choice!=6:

        #path = "C:/Users/Peter/Desktop/MIT/code/models/" + std_name + ".pickle"
        path = "./" + std_name + ".pickle"
        output = open(path, 'w+b')
        pickle.dump(std, output)
        output.close()

        #path = "C:/Users/Peter/Desktop/MIT/code/models/" + mean_name + ".pickle"
        path = "./" + mean_name + ".pickle"
        output = open(path, 'w+b')
        pickle.dump(mean, output)
        output.close()

        # serialize model to JSON
        model_json = trained_M.to_json()
        #with open("./models/" + model_name + ".json", "w") as json_file:
        with open("./" + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        #trained_M.save_weights("./models/" + model_name + ".h5")
        trained_M.save_weights("./" + model_name + ".h5")
        print("Saved model to disk")


def extract_descriptors_from_file_to_pickle(inputfile, outputfile, num_pos_sample=0):
    print("Working on: " + str(inputfile))
    print(" ")
    s_read_seq = time.time()
    if reduce_by_similarity == 1:
        if "_reduced" in inputfile:
            print("File already reduced to be maximum 90 percent identical! Clear reduce_by_similarity!")
            input()
        elif ".txt" in inputfile:
            name = inputfile.replace('.txt', '')
            file_to_reduce = open(inputfile)
            lines = file_to_reduce.readlines()
            if num_pos_sample != 0:
                lines = lines[:round(sc_1*num_pos_sample)]
            line_number = len(lines)
            file_to_reduce.close()
        elif ".fasta" in inputfile:
            name = inputfile.replace('.fasta', '')
            lines = IO.read_fasta_file(inputfile)
            lines = [str(line) for line in lines]
            if num_pos_sample != 0:
                lines = lines[:round(sc_1*num_pos_sample)]
            line_number = len(lines)
        else:
            print("Unknown file format! Use .fasta or .txt! Press CTRL-C to exit")
            input()

        out = name + "_reduced.txt"
        deleted = []
        sim_array = np.zeros((line_number, line_number))

        for i in list(range(line_number)):
            print("Doing line %d out of %d" %(i, line_number))
            string1 = lines[i].strip()
            for j in list(range(i+1, line_number)):
                #print(j)
                string2 = lines[j].strip()
                if similar(string1, string2) >= 0.9:
                    sim_array[i,j] = 1
                    sim_array[j,i] = 1

        while np.sum(np.sum(sim_array, 0)) != 0:
            sum_arr = np.sum(sim_array, 0)
            idx_to_be_deleted = np.argmax(sum_arr)
            sim_array = np.delete(sim_array, idx_to_be_deleted, 0)
            sim_array = np.delete(sim_array, idx_to_be_deleted, 1)
            deleted.append(lines[idx_to_be_deleted])
            del lines[idx_to_be_deleted]

        print("Deleted items:")
        [print(item) for item in deleted]

        f = open(out, "w+")
        for line in lines:
            f.write(line)
            f.write("\n")
        f.close()

        inputfile = out

    if ".txt" in inputfile:
        seqs = []
        with open(inputfile) as f:
            for line in f:
                seqs.append(line.strip()) #strip is important otherwis /n issue!
        inputfile = inputfile.replace("_reduced.txt", "")
    elif ".fasta" in inputfile:
        seqs = IO.read_fasta_file(inputfile)
        inputfile = inputfile.replace("_reduced.fasta", "")
    else:
        print("Unknown file format! Use .fasta or .txt! Press CTRL-C to exit")
        input()
    e_read_seq = time.time()
    print("Total time to read sequences: " + str(e_read_seq - s_read_seq))
    print(str(len(seqs)))
    chars = set('ARNDCQEGHILKMFPSTWYV')

    if inputfile in negfile:
        if num_pos_sample == 0:
            print ("Error, use Ctrl-C to quit")
            input()
        print(num_pos_sample)
        if num_pos_sample > len(seqs):
            print("Warning: Class balance may not be achieved! Click any button to accept or CTRL-C to exit")
            input()
        a = random.sample(range(1, len(seqs)), round(sc_2*num_pos_sample)) #if total_samples is big, you may want to divide total_samples (by 18) and round it
        newseqs = []
        i = 1
        for number in a:
            print(i)
            if len(seqs[number]) > minlength and all((c in chars) for c in seqs[number].upper()):
                newseqs.append(seqs[number])
                print(seqs[number])
                i = i+1
            if i > num_pos_sample:
                break
        if i < num_pos_sample:
            print("The negative set does not contain enough valid inputs to make the classifier balanced. Reduce downsampling! Use CTRL-C to quit!")
            input()
        seqs = newseqs
    #s_x_desc = time.time()
    dvecs = Manager().list()
    current_seq = Value('i', 1)
    dropped = 0
    lock = Lock()
    seqs = [s.upper() for s in seqs]
    mask = [all((c in chars) for c in s) and len(s) > minlength for s in seqs]
    seqs = list(compress(seqs, mask))
    total_samples = len(seqs)
    pool = Pool(numcores, initializer, (current_seq, dvecs, total_samples, lock))
    s_parallel = time.time()
    pool.map(thefunction, seqs)
    e_parallel = time.time()
    #pool.close()
    #pool.join()
    print("Total time to extract descriptors: " + str(e_parallel - s_parallel))
    if inputfile in posfile:
        num_pos_sample = len(dvecs)
        print("Number of positive samples: %d" %(num_pos_sample))
    #e_x_desc = time.time()
    #print("Total time to extract descriptors: " + str(e_x_desc - s_x_desc))
    print("Number of samples dropped due to meaningless characters: %d" %(dropped))

    y = dvecs._callmethod('__getitem__', (slice(1, total_samples+1),)) #THIS IS THE SOLUTION TO MAKE PICKLE WORK!!!!!!
    IO.serialize_descriptor_vector(y, o_file=outputfile)

    return num_pos_sample

def thefunction(seq):
    try:
        with lock:
            print("Extracting descriptors for sequence: " + str(current_seq.value) + "/" + str(total_samples))
            print(seq)
            print(" ")
            current_seq.value += 1
        dvec = FX.extract_named_descriptors_of_seq(seq)
        dvecs.append(dvec)
    except:
        print("A process just died.")


def initializer(*args):
    global current_seq, lock, dvecs, total_samples
    current_seq, dvecs, total_samples, lock = args

def get_performance_vals(y_test, classes):
    a = np.array(y_test)
    b = classes
    print("Predicted and actual classes\n")
    print(classes)
    print(a)
    tp = np.sum(np.multiply(a==1, b==1)) #TP
    fp = np.sum(np.multiply(b==1, a==0)) #FP
    tn = np.sum(np.multiply(a==0, b==0)) #TN
    fn = np.sum(np.multiply(a==1, b==0)) #FN

    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    print("True positive: %d, false positive: %d, true negative: %d, false negative: %d\n" %(tp,fp,tn,fn))

    mcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = 2*precision*recall/(precision+recall)
    accuracy = np.sum(a==b)/len(classes)

    print("%s: %2f" % ('MCC', mcc))
    print("%s: %.2f%%" % ('Accuracy', 100*accuracy))
    print("%s: %.2f" % ('F1 score', fscore))
    print("%s: %.2f" % ('Precision', precision))
    print("%s: %.2f" % ('Recall', recall))

    return mcc, precision, recall, fscore, accuracy

def do_kfolds(x, y):
    random.seed(seed)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accuracies = []
    fscores = []
    precisions = []
    recalls = []
    mccs = []
    x = np.array(x)
    y = np.array(y)
    print("Preparing training and label data...OK")
    print("")

    for train, test in kfold.split(x, y):

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        mean = np.mean(x_train, axis = 0)
        std = np.std(x_train, axis = 0)
        x_train -= mean
        eps = 10**-5
        std = std + eps
        x_train /= std

        x_test -= mean
        x_test /= std

        print("Training model on data...")
        s_training = time.time()
        M = trainer.build_sequential_model(rate = 0.3, shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)

        print("Classifying data...")
        s_classify = time.time()
        #scores = trained_M.(x_test)
        classes = trained_M.predict_classes(x_test)
        classes = np.array(classes)
        classes = classes.ravel()

        e_classify = time.time()
        print("Classifying data...OK, took: " + str((e_classify - s_classify)))

        mcc, accuracy, fscore, precision, recall = get_performance_vals(y_test, classes)

        mccs.append(mcc)
        accuracies.append(accuracy)
        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)

    print("MCC: %.2f (+/- %.2f)" % (np.mean(mccs), np.std(mccs)))
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (100*np.mean(accuracies), 100*np.std(accuracies)))
    print("F1 score: %.2f (+/- %.2f)" % (np.mean(fscores), np.std(fscores)))
    print("Precision: %.2f (+/- %.2f)" % (np.mean(precisions), np.std(precisions)))
    print("Recall: %.2f (+/- %.2f)" % (np.mean(recalls), np.std(recalls)))

    return trained_M, mean, std

def do_crossval(x,y):
    mccs = []
    rate_arr = [0.2, 0.3, 0.4, 0.5, 0.6] #dropout rate

    x, x_test, y, y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=seed,stratify=y)
    x_train, x_validate, y_train, y_validate = train_test_split(x,y,test_size = 0.25,train_size =0.75,random_state=seed,stratify=y)

    mean = np.mean(x_train, axis = 0)
    std = np.std(x_train, axis = 0)
    x_train -= mean
    eps = 10**-5
    std = std + eps
    x_train /= std

    x_validate -= mean
    x_validate /= std

    x_test -= mean
    x_test /= std

    for i in rate_arr:
        print("Training model on data...")
        s_training = time.time()
        M = trainer.build_sequential_model(rate = i, shape = x_train.shape[1])
        trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000) #set high to 500
        e_training = time.time()
        print("Training model on data...OK, took: " + str((e_training - s_training)))

        print("Classifying data...")
        s_classify = time.time()
        #scores = trained_M.predict(x_validate)
        classes = trained_M.predict_classes(x_validate)
        classes = np.array(classes)
        classes = classes.ravel()

        e_classify = time.time()
        print("Classifying data...OK, took: " + str((e_classify - s_classify)))

        mcc, accuracy, fscore, precision, recall = get_performance_vals(y_validate, classes)

        mccs.append(mcc)

    idx = np.argmax(mccs)
    best_rate = rate_arr[idx]
    print("Best dropout rate is %f" %(best_rate))

    print("Training model on data...")
    s_training = time.time()
    M = trainer.build_sequential_model(rate = best_rate, shape = x_train.shape[1])
    trained_M = trainer.fit_model_batch(M, x_train, y_train, num_epoch=2000)
    e_training = time.time()
    print("Training model on data...OK, took: " + str((e_training - s_training)))

    print("Classifying data...")
    s_classify = time.time()
    #scores = trained_M.predict(x_test)
    classes = trained_M.predict_classes(x_test)
    classes = np.array(classes)
    classes = classes.ravel()

    e_classify = time.time()
    print("Classifying data...OK, took: " + str((e_classify - s_classify)))

    print("Best dropout rate is %f" %(best_rate))
    mcc, accuracy, fscore, precision, recall = get_performance_vals(y_test, classes)

    return trained_M, mean, std


if __name__ == "__main__":

    #model = IO.deserialize_model("models/basic_sequential")
    model = None
    main(model)
