import re
import numpy as np
import numpy
import glob
import random
import pickle
from tqdm import tqdm
from numpy.core.fromnumeric import std
import mir_eval
from cfp import cfp_process
from tensorflow import keras

from constant import *
from loader import *
import essentia.standard as estd
from generator import create_data_generator
from loader import load_data, load_data_for_test  # TODO

from network.ftanet import create_model
from loader import get_CenFreq

DATA_PATH = '/mnt/sda1/genis/carnatic_melody_synthesis/resources/Saraga-Synth-Dataset/experiments'


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def est(output, CenFreq, time_arr):
    # output: (freq_bins, T)
    CenFreq[0] = 0
    est_time = time_arr
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]

    if len(est_freq) != len(est_time):
        new_length = min(len(est_freq), len(est_time))
        est_freq = est_freq[:new_length]
        est_time = est_time[:new_length]

    est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

    return est_arr


def melody_eval(ref, est):
    ref_time = ref[:, 0]
    ref_freq = ref[:, 1]

    est_time = est[:, 0]
    est_freq = est[:, 1]

    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    VR = output_eval['Voicing Recall'] * 100.0
    VFA = output_eval['Voicing False Alarm'] * 100.0
    RPA = output_eval['Raw Pitch Accuracy'] * 100.0
    RCA = output_eval['Raw Chroma Accuracy'] * 100.0
    OA = output_eval['Overall Accuracy'] * 100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr


def iseg(data):
    # data: (batch_size, freq_bins, seg_len)
    new_length = data.shape[0] * data.shape[-1]  # T = batch_size * seg_len
    new_data = np.zeros((data.shape[1], new_length))  # (freq_bins, T)
    for i in range(len(data)):
        new_data[:, i * data.shape[-1] : (i + 1) * data.shape[-1]] = data[i]
    return new_data


def evaluate(model, x_list, y_list, batch_size, filename=None):
    list_to_save = []
    avg_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size:]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size]
                length = batch_size

            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k]) 
            prediction = model.predict(X, length)
            preds.append(prediction)

        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)

        # ground-truth
        ref_arr = y
        time_arr = y[:, 0]
        
        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        #CenFreq = get_CenFreq(StartFreq=20, StopFreq=2048, NumPerOct=60)
        #CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=111)
        #CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=190)
        est_arr = est(preds, CenFreq, time_arr)

        # evaluates
        eval_arr = melody_eval(ref_arr, est_arr)
        list_to_save.append(eval_arr)
        avg_eval_arr += eval_arr
    
    avg_eval_arr /= len(x_list)
    # VR, VFA, RPA, RCA, OA
    if filename:
        with open('/mnt/sda1/genis/FTANet-melodic/file_lists/' + filename, 'wb') as f:
            pickle.dump(list_to_save, f)
    return avg_eval_arr


def get_est_arr(model, x_list, y_list, batch_size):
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        
        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j * batch_size:]
                length = x.shape[0] - j * batch_size
            else:
                X = x[j * batch_size: (j + 1) * batch_size]
                length = batch_size
            
            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k])
            prediction = model.predict(X, length)
            preds.append(prediction)
        
        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)
        
        # ground-truth
        
        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=20, StopFreq=2048, NumPerOct=60)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=111)
        # CenFreq = get_CenFreq(StartFreq=81, StopFreq=600, NumPerOct=190)
        est_arr = est(preds, CenFreq, y)
        
    # VR, VFA, RPA, RCA, OA
    return est_arr


def get_pitch_track(filename):
    print('Loading model...')
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(
        filepath='/mnt/sda1/genis/FTANet-melodic/model/baseline/OA/best_OA'
    ).expect_partial()
    print('Model loaded!')
    
    xlist = []
    timestamps = []
    # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
    feature, _, time_arr = cfp_process(filename, sr=8000, hop=80)
    print('feature', np.shape(feature))
    
    data = batchize_test(feature, size=128)
    xlist.append(data)
    timestamps.append(time_arr)
    
    print(np.shape(data), np.shape(time_arr))
    
    estimation = get_est_arr(model, xlist, timestamps, batch_size=16)

    save_pitch_track_to_dataset(
        '/mnt/sda1/genis/FTANet-melodic/data/experiment.txt',
        estimation[:, 0],
        estimation[:, 1]
    )


def evaluate_model():
    print('Loading model...')
    model_baseline = create_model(input_shape=IN_SHAPE)
    model_baseline.load_weights('/mnt/sda1/genis/FTANet-melodic/model/ftanet.h5')
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(
        filepath='/mnt/sda1/genis/FTANet-melodic/model/baseline/OA/best_OA'
        ).expect_partial()
    print('Model loaded!')
    
    xlist = []
    ylist = []
    #timestamps = []
    for chunk_id in tqdm(['2500']):
        
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        wav_file = DATA_PATH + '/audio/synth_mix_' + chunk_id + '.wav'
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
        print('feature', np.shape(feature))
    
        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        ref_arr = csv2ref(DATA_PATH + '/annotations/melody/synth_mix_' + chunk_id + '.csv')
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        print('pitch', np.shape(ref_arr_res))
    
        data = batchize_test(feature, size=128)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])
        #timestamps.append(time_arr)

    hola = evaluate(model, xlist, ylist, batch_size=16)
    hola2 = evaluate(model_baseline, xlist, ylist, batch_size=16)
    #estimation = get_est_arr(model, xlist, timestamps, batch_size=16)
    print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        hola[0], hola[1], hola[2], hola[3], hola[4]))
    print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        hola2[0], hola2[1], hola2[2], hola2[3], hola2[4]))
    
    #save_pitch_track_to_dataset(
    #    '/mnt/sda1/genis/FTANet-melodic/data/experiment.txt',
    #    estimation[:, 0],
    #    estimation[:, 1]
    #)
    
    
def test_model():
    print('Loading model...')
    #model_baseline = create_model(input_shape=IN_SHAPE)
    #model_baseline.load_weights('/mnt/sda1/genis/FTANet-melodic/model/ftanet.h5')
    model = create_model(input_shape=IN_SHAPE)
    model.load_weights(
        filepath='/mnt/sda1/genis/FTANet-melodic/model/baseline/OA/best_OA'
    ).expect_partial()
    print('Model loaded!')
    
    train_file = open('/mnt/sda1/genis/FTANet-melodic/file_lists/train.pkl', 'rb')
    train_list = pickle.load(train_file)

    test_file = open('/mnt/sda1/genis/FTANet-melodic/file_lists/eval.pkl', 'rb')
    test_list = pickle.load(test_file)
    
    data_files = glob.glob(DATA_PATH + '/audio/synth_mix*')
    data_idx = [x.split('/')[-1].replace('synth_mix_', '').replace('.wav', '') for x in data_files]
    
    testing_files = [x for x in data_idx if x not in train_list]
    testing_files = [x for x in testing_files if x not in test_list]
    
    #print(len(data_files))
    #print(len(train_list))
    #print(len(test_list))
    #print(len(testing_files))

    # Get unseen concerts during training
    testing_files.sort(key=natural_keys)

    xlist = []
    ylist = []
    #ylist_baseline = []
    # timestamps = []
    for chunk_id in tqdm(random.sample(testing_files[:500], 125)):
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        wav_file = DATA_PATH + '/audio/synth_mix_' + chunk_id + '.wav'
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
        print('feature', np.shape(feature))
    
        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        ref_arr = csv2ref(DATA_PATH + '/annotations/melody/synth_mix_' + chunk_id + '.csv')
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        print('pitch', np.shape(ref_arr_res))
    
        data = batchize_test(feature, size=128)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])
        #ylist_baseline.append(ref_arr[:, :])
        # timestamps.append(time_arr)

    hola = evaluate(model, xlist, ylist, batch_size=16, filename='ours_test.pkl')
    #hola2 = evaluate(model_baseline, xlist, ylist_baseline, batch_size=16, filename='baseline_2.pkl')
    # estimation = get_est_arr(model, xlist, timestamps, batch_size=16)
    print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        hola[0], hola[1], hola[2], hola[3], hola[4]))
    #print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
    #    hola2[0], hola2[1], hola2[2], hola2[3], hola2[4]))
    

def save_pitch_track_to_dataset(filename, est_time, est_freq):
    # Write txt annotation to file
    with open(filename, 'w') as f:
        for i, j in zip(est_time, est_freq):
            f.write("{}, {}\n".format(i, j))
    print('Saved with exit to {}'.format(filename))


# Just for test
if __name__ == '__main__':
    test_model()
    #get_pitch_track(
    #    '/mnt/sda1/genis/carnatic_melody_synthesis/resources/saraga_subset/Gopi Gopala Bala.mp3.mp3'
    #)
