import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data # train, val, or test datasets NOTE
        self.len = len(self.data) # e.g., 314, 
        self.subjects_dict = subjects_dict # {'train': ['FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA'], 'val': ['FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA'], 'test': ['FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA']}
        self.data_type = data_type # 'train', 'val', 'test'
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        # [8, 8] 单位矩阵, for 'vocaset'.
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(
                subject)]
        else:
            one_hot = self.one_hot_labels
        import ipdb; ipdb.set_trace()
        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

        # 1. audio, (85067,)
        # 2. vertice, (160, 15069) 
        # 3. template, (15069,)
        # 4. one_hot, array([0., 0., 0., 1., 0., 0., 0., 0.])
        # 5. file_name, 'FaceTalk_170915_00223_TA_sentence06.wav'

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path) # 'vocaset/wav'
    vertices_path = os.path.join(args.dataset, args.vertices_path) # 'vocaset/vertices_npy'
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(args.dataset, args.template_file) # 'vocaset/templates.pkl'
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1') # dict, keys=dict_keys(['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170904_03276_TA'])
    
    for r, ds, fs in os.walk(audio_path): # len(fs)=475
        for f in tqdm(fs):
            if f.endswith("wav"): # 'FaceTalk_170915_00223_TA_sentence21.wav'
                wav_path = os.path.join(r,f) # 'vocaset/wav/FaceTalk_170915_00223_TA_sentence21.wav', sr=22000
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000) # TODO 这里可以指定使用的audio的采样率; e.g., speech_array.shape=(69067,), 本来音频采样率是sr=22000，现在是按照16000读取的。
                input_values = np.squeeze(  
                        processor(speech_array,sampling_rate=16000).input_values)
                # 上面是对音频的预处理. output is input_values.shape=(69067,)
                key = f.replace("wav", "npy") # 'FaceTalk_170915_00223_TA_sentence21.npy'
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1]) # 'FaceTalk_170915_00223_TA'
                temp = templates[subject_id] # temp.shape = (5023, 3) 模板 NOTE
                data[key]["name"] = f # f = 'FaceTalk_170915_00223_TA_sentence21.wav'
                data[key]["template"] = temp.reshape((-1)) # temp from (5023, 3) to (15069,) 一帧模板
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy")) # 'vocaset/vertices_npy/FaceTalk_170915_00223_TA_sentence21.npy'
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(
                                vertice_path,allow_pickle=True)[::2,:] # 本来是259视频帧：(259, 15069) --> half --> 现在是只要130视频帧了：(130, 15069)
                        #due to the memory limit NOTE TODO can be updated...
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(
                                vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")] # 8, 'FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA'
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")] # 2, 'FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA'
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")] # 2, 'FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA'

    splits = {
            'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)}, # TODO why in this range? 

     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data)) # 314 40 39 for 'vocaset'
    return train_data, valid_data, test_data, subjects_dict
    # len(train_data) = 314, which is from 0 to 313
    # train_data[0] is a dict with keys = dict_keys(['audio', 'name', 'template', 'vertice'])
    # 1. train_data[0]['audio'].shape=(69067,)
    # 2. train_data[0]['name']='FaceTalk_170915_00223_TA_sentence21.wav'
    # 3. train_data[0]['template'].shape = (15069,)
    # 4. train_data[0]['vertice'].shape = (130, 15069)

    # 训练数据有8个主题；val数据有2个主题subjects；test数据有2个主题subjects. NOTE 
    # subjects_dict = {'train': ['FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA'], 'val': ['FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA'], 'test': ['FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA']}

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(
            dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(
            dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(
            dataset=test_data, batch_size=1, shuffle=False)
    return dataset # {'train': <torch.utils.data.dataloader.DataLoader object at 0x7f8c67ddd6d0>, 'valid': <torch.utils.data.dataloader.DataLoader object at 0x7f8c73905df0>, 'test': <torch.utils.data.dataloader.DataLoader object at 0x7f8c7b8b3610>} for 'vocaset'

if __name__ == "__main__":
    get_dataloaders()
    
