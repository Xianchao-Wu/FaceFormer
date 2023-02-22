import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path): # 'demo/result'
        os.makedirs(args.result_path)
    import ipdb; ipdb.set_trace()
    #build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, 
        '{}.pth'.format(args.model_name))))
    model = model.to(torch.device(args.device)) 
    # 可训练参数, [biwi] 108,487,646=108.5M; 全体参数, 112,688,094=112.7M
    # [vocaset] 92,215,197 = 92.2M 参数规模 for vocaset dataset.
    model.eval()
    
    # 面部模板文件，BIWI以及VOCASET分别有定义自己的模板
    template_file = os.path.join(args.dataset, args.template_path) 
    # 'BIWI/templates.pkl' ||| 'vocaset/templates.pkl'

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1') 
        # [BIWI] dict_keys(['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 
        # 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']) 面部face有8个点；嘴mouth有6个点. NOTE
        # [vocaset] 12 keys for vocaset, dict_keys(['FaceTalk_170904_00128_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170908_03277_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170904_03276_TA'])

    train_subjects_list = [i for i in args.train_subjects.split(" ")] 
    # [BIWI] 'F2 F3 F4 M3 M4 M5' -> ['F2', 'F3', 'F4', 'M3', 'M4', 'M5']
    # [vocaset] 8 keys as 'train': ['FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA']

    one_hot_labels = np.eye(len(train_subjects_list)) # (6, 6) I 单位矩阵 ||| (8, 8) for vocaset
    iter = train_subjects_list.index(args.condition) 
    # TODO args.condition='M3' 这个是啥意思? iter=3, mouth-3? for the 3-rd subject
    # 'vocaset', 'FaceTalk_170913_03279_TA', iter=5 for the 5-th subject

    one_hot = one_hot_labels[iter] # array([0., 0., 0., 1., 0., 0.]) ||| array([0., 0., 0., 0., 0., 1., 0., 0.])
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0])) # one_hot.shape = (6,) -> (1, 6) ||| array([[0., 0., 0., 0., 0., 1., 0., 0.]]) with shape = (1, 8)
    one_hot = torch.FloatTensor(one_hot).to(device=args.device) # [1, 6] in tensor format ||| torch.Size([1, 8])

    temp = templates[args.subject] # args.subject='M1', mouth 1; temp=[23370, 3], ||| 'FaceTalk_170809_00138_TA', (5023, 3) for 'vocaset'
             
    template = temp.reshape((-1)) # [23370, 3] -> 拍平 -> (70110,) ||| (15069,)
    template = np.reshape(template,(-1,template.shape[0])) # (1, 70110) ||| (1, 15069)
    template = torch.FloatTensor(template).to(device=args.device) # torch.Size([1, 70110]) ||| torch.Size([1, 15069])

    # 读取音频文件输入：
    wav_path = args.wav_path # 'demo/wav/test.wav'
    test_name = os.path.basename(wav_path).split(".")[0] # test_name = 'test'
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000) 
    # NOTE 重要，这里是按照采样率16k来读取音频文件！
    # 如果原始音频不是16k采样率，则重新采样到16k (背后操作), 
    # 目前的test.wav的采样率是44100的，所以会自动下采样到16000 = 16k. 
    # speech_array.shape=(184274,), sampling_rate=16k. TODO

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values) 
    # (184274,)

    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0])) # (1, 184274)
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device) 
    # torch.Size([1, 184274])

    import ipdb; ipdb.set_trace() # NOTE important here!
    prediction = model.predict(audio_feature, template, one_hot) 
    # audio_feature.shape=[1, 184274], 
    # template.shape=[1, 70110], ||| [1, 15069] for 'vocaset'
    # one_hot=tensor([[0., 0., 0., 1., 0., 0.]], device='cuda:0') ||| torch.Size([1, 8])

    # OUTPUT: prediction.shape = [1, 278, 70110] ||| [1, 278, 15069]
    # 278 = frame.num/2, 每两帧语音对应到一个包括了70110个点的“图片”
    # 70110 = (23370, 3)

    prediction = prediction.squeeze() # (seq_len, V*3), [1, 278, 70110] to [287, 70110]

    # NOTE
    # 这是把预测结果保存到一个具体的文件：
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

# The implementation of rendering is borrowed from 
# VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    import ipdb; ipdb.set_trace()
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, 
            faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, 
            material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], 
                viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence(args):
    import ipdb; ipdb.set_trace()

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    predicted_vertices_path = os.path.join(args.result_path,test_name+".npy")
    if args.dataset == "BIWI":
        template_file = os.path.join(args.dataset, 
                args.render_template_path, "BIWI.ply")
    elif args.dataset == "vocaset":
        template_file = os.path.join(args.dataset, 
                args.render_template_path, "FLAME_sample.ply")
         
    print("rendering: ", test_name)
                 
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, 
            cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(args,render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name+"_"+args.subject+"_condition_"+args.condition
    import ipdb; ipdb.set_trace()
    video_fname = os.path.join(output_path, file_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)

def main():
    import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser(
        description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="biwi")
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--fps", 
            type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", 
            type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", 
            type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", 
            type=int, default=23370*3, 
            help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", 
            type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", 
            type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", 
            type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", 
            type=str, default="M3", 
            help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", 
            type=str, default="M1", 
            help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", 
            type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", 
            type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", 
            type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()   

    import ipdb; ipdb.set_trace()
    test_model(args)

    import ipdb; ipdb.set_trace()
    render_sequence(args)

if __name__=="__main__":
    main()
