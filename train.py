import os
import csv
import random
import pickle
import argparse
import time
from tqdm import tqdm
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.metrics import categorical_accuracy

from constant import *
from generator import create_data_generator
from loader import load_data, load_data_for_test  # TODO
from evaluator import evaluate

from network.ftanet import create_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 指定输出的模型的路径
#parser = argparse.ArgumentParser()
#parser.add_argument("model_file", type=str, help="model file")
#checkpoint_model_file = parser.parse_args().model_file
fp = '/mnt/sda1/genis/carnatic_melody_synthesis/resources/Saraga-Synth-Dataset/experiments'
checkpoint_best_OA = '/mnt/sda1/genis/FTANet-melodic/model/baseline/OA/best_OA'
checkpoint_best_loss = '/mnt/sda1/genis/FTANet-melodic/model/baseline/loss/best_loss'
checkpoint_best_RPA = '/mnt/sda1/genis/FTANet-melodic/model/baseline/RPA/best_RPA'

list_folder = '/mnt/sda1/genis/FTANet-melodic/file_lists'


#log_file_name = 'log/log-train-{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))
#log_file_name = checkpoint_model_file.replace('model/', 'log/').replace('.h5', '.log')
#log_file = open(log_file_name, 'wb')

non_silent_tracks = []
print(len(os.listdir(os.path.join(fp, 'annotations', 'melody'))))
for track_path in tqdm(os.listdir(os.path.join(fp, 'annotations', 'melody'))):
    with open(os.path.join(fp, 'annotations', 'melody', track_path)) as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        for row in reader:
            pitch_value = float(row[1].replace(' ', ''))
            if float(pitch_value) > 0.01:
                non_silent_tracks.append(track_path.split('_')[-1].replace('.csv', ''))
                break

# Getting training split
train_portion = int(len(non_silent_tracks) * 0.65)
train_tracks = non_silent_tracks[:train_portion]
train_ids = random.sample(train_tracks, 950)  # Get 950 samples
train_ids = [x for x in train_ids if '9999999' not in x]
train_ids = [x for x in train_ids if 'xxxxx' not in x]

# Getting valid split
val_tracks = non_silent_tracks[train_portion:]
val_ids = random.sample(val_tracks, 25) + random.sample(train_tracks, 25)
val_ids = [x for x in val_ids if '9999999' not in x]
val_ids = [x for x in val_ids if 'xxxxx' not in x]

# Store list of selected files to reproduce the results
with open(list_folder + '/train.pkl', 'wb') as f:
    pickle.dump(train_ids, f)
with open(list_folder + '/eval.pkl', 'wb') as f:
    pickle.dump(val_ids, f)

##--- 加载数据 ---##
# x: (n, freq_bins, time_frames, 3) extract from audio by cfp_process
# y: (n, freq_bins+1, time_frames) from ground-truth
train_x, train_y, train_num = load_data(
    data_file='./',
    track_list=train_ids,
    seg_len=SEG_LEN
)
valid_x, valid_y = load_data_for_test(
    data_file='./',
    track_list=val_ids,
    seg_len=SEG_LEN
)

##--- Data Generator ---##
print('\nCreating generators...')
train_generator = create_data_generator(train_x, train_y, batch_size=BATCH_SIZE)

##--- 网络 ---##
print('\nCreating model...')

model = create_model(input_shape=IN_SHAPE)
model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=LR)))
# model.summary()

##--- 开始训练 ---##
print('\nTaining...')
print('params={}'.format(model.count_params()))

epoch, iteration = 0, 0
best_OA, best_epoch, best_RPA, best_loss = 0, 0, 0, 10000
mean_loss = 0
time_start = time.time()
while epoch < EPOCHS:
    iteration += 1
    print('Epoch {}/{} - {:3d}/{:3d}'.format(
        epoch+1, EPOCHS, iteration%(train_num//BATCH_SIZE), train_num//BATCH_SIZE), end='\r')
    # 取1个batch数据
    X, y = next(train_generator)
    # 训练1个iteration
    loss = model.train_on_batch(X, y)
    mean_loss += loss
    # 每个epoch输出信息
    if iteration % (train_num//BATCH_SIZE) == 0:
        # train meassage
        epoch += 1
        traintime = time.time() - time_start
        mean_loss /= train_num//BATCH_SIZE
        print('', end='\r')
        print('Epoch {}/{} - {:.1f}s - loss {:.4f}'.format(epoch, EPOCHS, traintime, mean_loss))
        # valid results
        avg_eval_arr = evaluate(model, valid_x, valid_y, BATCH_SIZE)
        
        # save best OA model
        if avg_eval_arr[-1] > best_OA:
            best_OA = avg_eval_arr[-1]
            best_epoch = epoch
            model.save_weights(
                filepath=checkpoint_best_OA,
                overwrite=True,
                save_format='tf'
            )
            print('Saved to ' + checkpoint_best_OA)

        # save best loss model
        if mean_loss <= best_loss:
            best_loss = mean_loss
            model.save_weights(
                filepath=checkpoint_best_loss,
                overwrite=True,
                save_format='tf'
            )
            print('Saved to ' + checkpoint_best_loss)

        # save best RPA model
        if avg_eval_arr[2] > best_RPA:
            best_RPA = avg_eval_arr[2]
            model.save_weights(
                filepath=checkpoint_best_RPA,
                overwrite=True,
                save_format='tf'
            )
            print('Saved to ' + checkpoint_best_RPA)
        print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
        # early stopping
        if epoch - best_epoch >= PATIENCE:
            print('Early stopping with best OA {:.2f}%'.format(best_OA))
            break
            
        # initialization
        mean_loss = 0
        time_start = time.time()
