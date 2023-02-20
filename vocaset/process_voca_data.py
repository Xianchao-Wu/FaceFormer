import os
import argparse
import cv2
import pickle
import sys
import numpy as np
from scipy.io import wavfile

def load_data(args):
    import ipdb; ipdb.set_trace()
    face_vert_mmap = np.load(args.verts_path, mmap_mode='r+') # 'data_verts.npy', 14GB, (123341, 5023, 3)
    raw_audio = pickle.load(open(args.raw_audio_path, 'rb'), encoding='latin1') # dict, with 12 keys: dict_keys(['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170904_03276_TA'])
    # NOTE 
    data2array_verts = pickle.load(open(args.data2array_verts_path, 'rb'))
    # dict, with 12 keys, dict_keys(['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170904_03276_TA'])

    import ipdb; ipdb.set_trace()
    return face_vert_mmap, raw_audio, data2array_verts

def generate_vertices_npy(args, face_vert_mmap, data2array_verts):
    # face_vert_mmap.shape = (123341, 5023, 3)
    # dict with 12 keys, dict_keys(['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170904_03276_TA']) 

    import ipdb; ipdb.set_trace()
    if not os.path.exists(args.vertices_npy_path): # 'vertices_npy'
        os.makedirs(args.vertices_npy_path)
    for sub in data2array_verts.keys(): # 'FaceTalk_170904_00128_TA'
        for seq in data2array_verts[sub].keys(): # dict_keys(['sentence37', 'sentence29', 'sentence28', 'sentence30', 'sentence31', 'sentence32', 'sentence33', 'sentence34', 'sentence35', 'sentence18', 'sentence19', 'sentence16', 'sentence17', 'sentence14', 'sentence15', 'sentence12', 'sentence13', 'sentence10', 'sentence11', 'sentence40', 'sentence23', 'sentence22', 'sentence27', 'sentence26', 'sentence25', 'sentence24', 'sentence09', 'sentence08', 'sentence21', 'sentence20', 'sentence05', 'sentence04', 'sentence07', 'sentence06', 'sentence01', 'sentence03', 'sentence02', 'sentence38', 'sentence36', 'sentence39'])
            vertices_npy_name = sub + "_" + seq # e.g., 'FaceTalk_170904_00128_TA_sentence37'
            vertices_npy = []
            for frame, array_idx in data2array_verts[sub][seq].items():
                vertices_npy.append(face_vert_mmap[array_idx]) # array_idx=30257, face_vert_mmap[30257].shape = (5023, 3)
            vertices_npy = np.array(vertices_npy).reshape(-1,args.vertices_dim) # [203, 15069]
            np.save(os.path.join(args.vertices_npy_path,vertices_npy_name) ,vertices_npy) # 'vertices_npy/FaceTalk_170904_00128_TA_sentence37'

    import ipdb; ipdb.set_trace()
    print('done') # 得到的是：'vertices_npy' path下的478多个文件

def generate_wav(args,raw_audio):
    import ipdb; ipdb.set_trace()
    if not os.path.exists(args.wav_path):
        os.makedirs(args.wav_path)
    for sub in raw_audio.keys():
        for seq in raw_audio[sub].keys():
            wav_name = sub + "_" + seq 
            wavfile.write(os.path.join(args.wav_path, wav_name+'.wav'), raw_audio[sub][seq]['sample_rate'], raw_audio[sub][seq]['audio'])        

    import ipdb; ipdb.set_trace()
    print('done') # 得到的是: 'wav' path下的475个文件

def main():
    import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--verts_path", type=str, default="data_verts.npy")
    parser.add_argument("--vertices_npy_path", type=str, default="vertices_npy")
    parser.add_argument("--vertices_dim", type=int, default=5023*3)
    parser.add_argument("--raw_audio_path", type=str, default='raw_audio_fixed.pkl')
    parser.add_argument("--wav_path", type=str, default='wav')
    parser.add_argument("--data2array_verts_path", type=str, default='subj_seq_to_idx.pkl')
    args = parser.parse_args()
    # Namespace(data2array_verts_path='subj_seq_to_idx.pkl', raw_audio_path='raw_audio_fixed.pkl', vertices_dim=15069, vertices_npy_path='vertices_npy', verts_path='data_verts.npy', wav_path='wav')
    
    import ipdb; ipdb.set_trace()
    face_vert_mmap, raw_audio, data2array_verts = load_data(args) # 读取三个文件!

    import ipdb; ipdb.set_trace()
    generate_vertices_npy(args, face_vert_mmap, data2array_verts)

    import ipdb; ipdb.set_trace()
    generate_wav(args, raw_audio)

if __name__ == '__main__':
    main()
