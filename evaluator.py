import glob
import random
import pickle

from constant import *
from loader import *
import essentia.standard as estd

from network.ftanet import create_model
from loader import get_CenFreq


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


def melody_eval(ref, est, cent_tolerance=50):
    ref_time = ref[:, 0]
    ref_freq = ref[:, 1]

    est_time = est[:, 0]
    est_freq = est[:, 1]

    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq, cent_tolerance=cent_tolerance)
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


def evaluate(model, x_list, y_list, batch_size, cent_tolerance=50, filename=None):
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
        #CenFreq = get_CenFreq(StartFreq=100, StopFreq=600, NumPerOct=124)
        est_arr = est(preds, CenFreq, time_arr)

        # evaluates
        eval_arr = melody_eval(ref_arr, est_arr, cent_tolerance=cent_tolerance)
        list_to_save.append(eval_arr)
        avg_eval_arr += eval_arr

    if filename:
        with open('/mnt/sda1/genis/FTANet-melodic/file_lists/' + filename, 'wb') as f:
            pickle.dump(file=f, obj=list_to_save)
    
    avg_eval_arr /= len(x_list)
    # VR, VFA, RPA, RCA, OA
    
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
    
    estimation = get_est_arr(model, xlist, timestamps, batch_size=16)

    save_pitch_track_to_dataset(
        '/mnt/sda1/genis/FTANet-melodic/data/experiment.txt',
        estimation[:, 0],
        estimation[:, 1]
    )
    
    
def test_model(file_list, save_name=''):
    print('Loading synth Carnatic model...')
    model_OA = create_model(input_shape=IN_SHAPE)
    model_OA.load_weights(
        filepath='/home/genis/FTANet-melodic/model/synth_more_data_baseline/OA'
    ).expect_partial()

    model_non_synth = create_model(input_shape=IN_SHAPE)
    model_non_synth.load_weights(
        filepath='/home/genis/FTANet-melodic/model/more_resolution/best_OA'
    ).expect_partial()

    model_western = create_model(input_shape=IN_SHAPE)
    model_western.load_weights(
        filepath='/home/genis/FTANet-melodic/model/western/OA'
    ).expect_partial()

    xlist = []
    ylist = []

    for wav_file in tqdm(file_list):
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
        print('feature', np.shape(feature))
    
        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        ref_arr = csv2ref(wav_file.replace('.wav', '.csv').replace('audio', 'annotations/melody'))
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        print('pitch', np.shape(ref_arr_res))

        data = batchize_test(feature, size=128)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])

    #scores_bl = evaluate(model_baseline, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
    scores_synth = evaluate(
        model_OA, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
    scores_non_synth = evaluate(
        model_non_synth, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
    scores_western = evaluate(
        model_western, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)

    print('CARNATIC MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        scores_synth[0], scores_synth[1], scores_synth[2], scores_synth[3], scores_synth[4]))
    
    print('NON-SYNTH MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
         scores_non_synth[0], scores_non_synth[1], scores_non_synth[2], scores_non_synth[3], scores_non_synth[4]))

    print('WESTERN MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
         scores_western[0], scores_western[1], scores_western[2], scores_western[3], scores_western[4]))

    return scores_synth


def test_model_on_MIREX05(file_list, artist_name='hola'):
    print('Loading synth Carnatic model...')
    model_OA = create_model(input_shape=IN_SHAPE)
    model_OA.load_weights(
        filepath='/home/genis/FTANet-melodic/model/western/OA',
    ).expect_partial()

    # print('Loading baseline Western music model...')
    # model_baseline = create_model(input_shape=IN_SHAPE)
    # model_baseline.load_weights('/mnt/sda1/genis/FTANet-melodic/model/ftanet.h5')
    
    print('Models loaded!')
    
    xlist = []
    ylist = []
    
    for wav_file in tqdm(file_list):
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
        print('feature', np.shape(feature))
        
        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        ref_arr = txt2ref_tabs(wav_file.replace('.wav', 'REF.txt'))
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        print('pitch', np.shape(ref_arr_res))
        
        data = batchize_test(feature, size=128)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])

    # scores_bl = evaluate(model_baseline, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
    scores_synth = evaluate(
        model_OA, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
    # scores_baseline = evaluate(
    #    model_baseline, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None
    # )
    print('CARNATIC MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        scores_synth[0], scores_synth[1], scores_synth[2], scores_synth[3], scores_synth[4]))
    # print('WESTERN MUSIC MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
    #    scores_baseline[0], scores_baseline[1], scores_baseline[2], scores_baseline[3], scores_baseline[4]))
    
    return scores_synth


def test_model_on_ADC(file_list, artist_name='hola'):
    print('Loading synth Carnatic model...')
    model_OA = create_model(input_shape=IN_SHAPE)
    model_OA.load_weights(
        filepath='/home/genis/FTANet-melodic/model/western/OA',
    ).expect_partial()
    
    # print('Loading baseline Western music model...')
    # model_baseline = create_model(input_shape=IN_SHAPE)
    # model_baseline.load_weights('/mnt/sda1/genis/FTANet-melodic/model/ftanet.h5')
    
    print('Models loaded!')
    
    xlist = []
    ylist = []
    scores_synth = []
    
    arranged_filelist = [x for x in file_list if 'daisy' in x] + \
                        [x for x in file_list if 'opera' in x] + \
                        [x for x in file_list if 'pop' in x]
    
    for wav_file in tqdm(arranged_filelist):
        ## Load cfp features (3, 320, T)
        # feature = np.load(data_folder + 'cfp/' + fname + '.npy')
        feature, _, time_arr = cfp_process(wav_file, sr=8000, hop=80)
        print('feature', np.shape(feature))
        
        ## Load f0 frequency
        # pitch = np.loadtxt(data_folder + 'f0ref/' + fname + '.txt')
        ref_arr = txt2ref_spaces(wav_file.replace('.wav', 'REF.txt'))
        times, pitch = resample_melody(ref_arr, np.shape(feature)[-1])
        ref_arr_res = np.concatenate((times[:, None], pitch[:, None]), axis=1)
        print('pitch', np.shape(ref_arr_res))
        
        data = batchize_test(feature, size=128)
        xlist.append(data)
        ylist.append(ref_arr_res[:, :])
        
        # scores_bl = evaluate(model_baseline, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
        scores_synth = evaluate(
            model_OA, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None)
        # scores_baseline = evaluate(
        #    model_baseline, xlist, ylist, batch_size=16, cent_tolerance=50, filename=None
        # )
        print('CARNATIC MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            scores_synth[0], scores_synth[1], scores_synth[2], scores_synth[3], scores_synth[4]))
        # print('WESTERN MUSIC MODEL\nVR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
        #    scores_baseline[0], scores_baseline[1], scores_baseline[2], scores_baseline[3], scores_baseline[4]))
    
    return scores_synth


def save_pitch_track_to_dataset(filename, est_time, est_freq):
    # Write txt annotation to file
    with open(filename, 'w') as f:
        for i, j in zip(est_time, est_freq):
            f.write("{}, {}\n".format(i, j))
    print('Saved with exit to {}'.format(filename))


def predict_melodia(filepath, filename='', evaluate=True):
    melodia_extractor = estd.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    
    audio = estd.EqloudLoader(filename=filepath, sampleRate=44100)()
    est_freq, _ = melodia_extractor(audio)
    est_freq = np.append(est_freq, 0.0)
    est_time = np.linspace(0.0, len(audio) / 44100, len(est_freq))
    est_arr_melodia = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)
    
    if filename:
        with open('/mnt/sda1/genis/FTANet-melodic/file_lists/' + filename, 'w') as f:
            for i, j in zip(est_arr_melodia[:, 0], est_arr_melodia[:, 1]):
                f.write(str(i) + ',' + str(j) + '\n')
        print(filename, 'Saved correctly')
        
    
    if evaluate is not None:

        filepath_evaluate = filepath.replace('.wav', '.csv').replace('audio', 'annotations/melody')
        if 'csv' in filepath_evaluate:
            ycsv = pd.read_csv(filepath_evaluate, names=["time", "freq"])
            gtt = ycsv['time'].values
            gtf = ycsv['freq'].values
            ref_arr = np.concatenate((gtt[:, None], gtf[:, None]), axis=1)
        elif 'txt' in filepath_evaluate:
            ref_arr = np.loadtxt(filepath_evaluate)
        else:
            print("Error: Wrong type of ground truth. The file must be '.txt' or '.csv' ")
            return None
        eval_arr_melodia = melody_eval(ref_arr, est_arr_melodia, cent_tolerance=50)
    
    return eval_arr_melodia


def evaluate_melodia(list_tracks):
    list_to_save = []
    eval_melodia = np.array([0, 0, 0, 0, 0], dtype='float16')
    max_OA, min_OA = 0, 100
    for i in tqdm(list_tracks):
        # eval_iter = predict(i, model_type, output_dir, gpu_index, evaluate, mode)
        eval_iter = predict_melodia(i, filename='experiment_3.txt')
        list_to_save.append(eval_iter)
        eval_melodia += eval_iter
        if eval_iter[4] > max_OA:
            max_OA = eval_iter[4]
        if eval_iter[4] < min_OA:
            min_OA = eval_iter[4]
    
    eval_melodia = eval_melodia / len(list_tracks)
    print('FINAL ESTIMATION MELODIA \n AVG | VR: {:.2f}% VFA: {:.2f}% RPA: {:.2f}% RCA: {:.2f}% OA: {:.2f}%'.format(
        eval_melodia[0], eval_melodia[1], eval_melodia[2], eval_melodia[3], eval_melodia[4])
    )


def select_vocal_track(ypath, lpath):
    ycsv = pd.read_csv(ypath, names=["time", "freq"])
    gt0 = ycsv['time'].values
    gt0 = gt0[:, np.newaxis]
    
    gt1 = ycsv['freq'].values
    gt1 = gt1[:, np.newaxis]
    
    z = np.zeros(gt1.shape)
    
    f = open(lpath, 'r')
    lines = f.readlines()
    
    for line in lines:
        
        if 'start_time' in line.split(',')[0]:
            continue
        st = float(line.split(',')[0])
        et = float(line.split(',')[1])
        sid = line.split(',')[2]
        for i in range(len(gt1)):
            if st < gt0[i, 0] < et and 'singer' in sid:
                z[i, 0] = gt1[i, 0]
    
    gt = np.concatenate((gt0, z), axis=1)
    return gt
 

def get_files_to_test(fp, artist, artists_to_track_mapping):
    # Get track to train
    tracks_to_test = artists_to_track_mapping[artist]

    # Get filenames to train
    files_to_test = []
    for track in tracks_to_test:
        files_to_test.append(fp + 'audio/' + track + '.wav')
    
    return files_to_test


# Run python3 evaluation.py to execute the evaluation code
if __name__ == '__main__':
    fp_synth = '/home/genis/Saraga-Carnatic-Melody-Synth/'
    fp_hindustani = '/home/genis/Hindustani-Synth-Dataset/'

    dataset_filelist_synth = glob.glob(fp_synth + 'audio/*.wav')
    with open(fp_synth + 'artists_to_track_mapping.pkl', 'rb') as map_file:
        artists_to_track_mapping = pickle.load(map_file)
    
    mahati_test = get_files_to_test(fp_synth, 'Mahati', artists_to_track_mapping)
    sumithra_test = get_files_to_test(fp_synth, 'Sumithra Vasudev', artists_to_track_mapping)
    modhumudi_test = get_files_to_test(fp_synth, 'Modhumudi Sudhakar', artists_to_track_mapping)
    chertala_test = get_files_to_test(fp_synth, 'Cherthala Ranganatha Sharma', artists_to_track_mapping)
    test_carnatic_list = [mahati_test, sumithra_test, modhumudi_test, chertala_test]
    testing_carnatic = []
    for i in test_carnatic_list:
        testing_carnatic += random.sample(i, 50)
        
    # Store testing set of Carnatic-Synth
    with open('./file_lists/Carnatic-Synth-test.txt') as f:
        for i in testing_carnatic:
            f.write(i + '\n')
        f.close()
    # Evaluate Carnatic-Synth set
    carnatic_synth_scores = test_model(testing_carnatic)
    
    # Parse hindustani testing set
    hindustani_testing_files = glob.glob(fp_hindustani + 'audio/*.wav')
    hindustani_testing_recordings = ['Deepki', 'Raag_Jog', 'Raag_Dhani', 'Todi', 'Malkauns', 'Piloo']
    testing_hindustani = []
    for i in hindustani_testing_recordings:
        testing_hindustani += [x for x in hindustani_testing_files if i in x]

    # Evaluate Hindustani-Synth set
    hindustani_synth = test_model(testing_hindustani)

    # Evaluate model on ADC2004
    adc_filelist = glob.glob(
        '/home/genis/FTANet-melodic/eval_datasets/ADC2004/*.wav'
    )
    scores_adc = test_model_on_ADC(adc_filelist)

    # Evaluate model on MIREX05
    mirex05_filelist = glob.glob(
        '/home/genis/FTANet-melodic/eval_datasets/MIREX05/*.wav'
    )
    scores_mirex05 = test_model_on_MIREX05(mirex05_filelist)
