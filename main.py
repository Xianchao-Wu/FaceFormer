import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, 
        dev_loader, model, optimizer, criterion, epoch=100):
    import ipdb; ipdb.set_trace()

    save_path = os.path.join(args.dataset, args.save_path) # 'vocaset/save'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")] 
    # ['FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 
    # 'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA', 
    # 'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA', 
    # 'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA'] -> 8 elements! NOTE
    iteration = 0

    for e in range(epoch+1): # 101=epoch+1
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        
        # 对一个batch的数据循环：
        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio = audio.to(device='cuda') # [1, 85067], 85067/16000=5.32秒的音频
            vertice = vertice.to(device="cuda") 
            # torch.Size([1, 160, 15069]) 视频中有160帧

            template = template.to(device="cuda") # torch.Size([1, 15069]), 视频模板
            one_hot = one_hot.to(device="cuda") 
            # one_hot = tensor([[0., 0., 0., 1., 0., 0., 0., 0.]]), 
            # -> shape=[1, 8]; 这是对一个subject[主题]的one-hot的表示. NOTE
            # file_name = ('FaceTalk_170904_03276_TA_sentence10.wav',)

            import ipdb; ipdb.set_trace() # NOTE
            # 调用模型的forward方法，获取loss 损失 # NOTE
            loss = model(audio, template,  
                    vertice, one_hot, criterion, teacher_forcing=False)


            import ipdb; ipdb.set_trace()
            loss.backward()

            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                    "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), 
                        iteration, np.mean(loss_log)))

        import ipdb; ipdb.set_trace()
        # validation, NOTE 走验证集合
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio = audio.to(device='cuda')
            vertice = vertice.to(device="cuda")
            template = template.to(device="cuda")
            one_hot_all = one_hot_all.to(device="cuda")

            train_subject = "_".join(file_name[0].split("_")[:-1])

            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
        
        import ipdb; ipdb.set_trace()
        current_loss = np.mean(valid_loss_log)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), 
                    os.path.join(
                        save_path,'{}_loss{}_model.pth'.format(e, current_loss)
                    )
            )

        import ipdb; ipdb.set_trace()
        print("epcoh: {}, current loss:{:.7f}".format(e + 1, current_loss))    
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    import ipdb; ipdb.set_trace()
    
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, 
        '{}_model.pth'.format(epoch))))

    model = model.to(torch.device("cuda"))
    model.eval()
   
    import ipdb; ipdb.set_trace()
    # 对测试集合的每个batch进行循环：
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio = audio.to(device='cuda')
        vertice = vertice.to(device="cuda")
        template = template.to(device="cuda")
        one_hot_all = one_hot_all.to(device="cuda")

        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, 
                file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"),
                prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, 
                    file_name[0].split(".")[0] + \
                            "_condition_"+condition_subject+".npy"), 
                    prediction.detach().cpu().numpy())
         
def count_parameters(model):
    # 计算模型中的“可训练参数”的个数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    import ipdb; ipdb.set_trace()

    # 下边的代码，确保每次启动train的时候，数据的顺序是一样的，方便debug
    seed=666
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True


    parser = argparse.ArgumentParser(
        description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, 
            default="vocaset", help='vocaset or BIWI')

    parser.add_argument("--vertice_dim", type=int, 
            default=5023*3, 
            help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')

    parser.add_argument("--feature_dim", type=int, 
            default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, 
            default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, 
            default= "wav", help='path of the audio signals')

    parser.add_argument("--vertices_path", type=str, 
            default="vertices_npy", help='path of the ground truth')

    parser.add_argument("--gradient_accumulation_steps", 
            type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, 
            default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", 
            type=str, default="templates.pkl", 
            help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, 
            default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, 
            default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, 
            default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA"
       " FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, 
            default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, 
            default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")

    import ipdb; ipdb.set_trace()
    args = parser.parse_args() 
    # Namespace(dataset='vocaset', device='cuda', feature_dim=64, 
    # gradient_accumulation_steps=1, lr=0.0001, max_epoch=100, period=30, 
    # result_path='result', save_path='save', template_file='templates.pkl', 
    # test_subjects='FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA', 
    # train_subjects='FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA 
    # FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA 
    # FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA 
    # FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA', 

    # val_subjects='FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA', 
    # vertice_dim=15069, vertices_path='vertices_npy', wav_path='wav')

    import ipdb; ipdb.set_trace()
    #build model
    model = Faceformer(args)
    
    import ipdb; ipdb.set_trace()
    print("model parameters: ", count_parameters(model)) 
    # 92,215,197=92M for 'vocaset' config; 如果是所有的参数：96,415,645. 
    # 即7层TCN in w2v model不需要训练.

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    import ipdb; ipdb.set_trace()
    dataset = get_dataloaders(args)
    # {'train': <torch.utils.data.dataloader.DataLoader object at 0x7fe1b1d1a160>,
    # 'valid': <torch.utils.data.dataloader.DataLoader object at 0x7fe1b1d6da00>, 
    # 'test': <torch.utils.data.dataloader.DataLoader object at 0x7fe1b458a1c0>} 
    # -> 'train', 'valid', 'test' 一共三种集合，区分
    # loss
    import ipdb; ipdb.set_trace()
    criterion = nn.MSELoss()

    # Train the model
    import ipdb; ipdb.set_trace()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
        model.parameters()), lr=args.lr) # args.lr=0.0001

    import ipdb; ipdb.set_trace() # NOTE 开始训练了...
    model = trainer(args, dataset["train"], 
        dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)
    # args.max_epoch=100, MSELoss=criterion,  
    import ipdb; ipdb.set_trace()
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()
