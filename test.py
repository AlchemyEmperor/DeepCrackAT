from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrackAT import DeepCrack
#from model.deepc import DeepCrack
#from model.dc_zmlp import DeepCrack
#from model.dc_smlp import DeepCrack
#from model.d1conv import DeepCrack
#from model.wcbam import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import dice_loss
from visdom import Visdom
import time
from metrics import iou_score

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# checkpoints/DeepCrack_Masonry_FT1/checkpoints/DeepCrack_CT260_FT1.pth
# test_data_path='data/Masonry/'

# test_data_path='data/CrackTree260/CrackTree260_Test_example.txt',
#          save_path='deepcrack_results_CT260/',
#          pretrained_model='checkpoints/DeepCrack_CT260_FT1/checkpoints/***.pth'

# test_data_path='data/Masonry/test_example.txt',
#          save_path='deepcrack_results2/',
#          pretrained_model='checkpoints/DeepCrack_Masonry_FT1/checkpoints/***.pth'

# test_data_path='data/Masonry/test_example.txt'

def test(test_data_path='data/Masonry_val/test.txt',
         save_path='results/Masonry/DeepCrackAT/',
         pretrained_model='checkpoints/Masonry/DeepCrackAT/checkpoints/DeepCrack_Masonry_FT1_BestModel.pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #测试数据路径获取以及Dataloader调用
    precision_score = 0
    accuracy_score = 0
    f1_score = 0
    recall_score = 0
    MIoU = 0
    k = 0
    TestTime = 0
    image_size = 224

    test_pipline = dataReadPip(transforms=None)

    test_list = readIndex(test_data_path)

    test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0, drop_last=False)
                                                            #num_workers表示进程数
    # -------------------- build trainer --------------------- #
    #使用GPU的Cuda加速
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    #利用之前训练好的网络权重进行测试
    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()

    total = sum([param.nelement() for param in model.parameters()])

    #测试的时候不需要保存梯度
    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                device)

            tt = test_target.detach().cpu().numpy()
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            save_pred = torch.zeros((224, 224))
            #特别注意 这里我们要保存的是二值图而不是灰度图 所以用torch.round进行转化
            test_pred = torch.round(test_pred)
            save_pred[:224, :] = test_pred
            #save_pred[224:, :] = lab.cpu().squeeze()
            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255

            tt = test_target.detach().cpu().numpy()

            cv2.imwrite(save_name, save_pred.astype(np.uint8))


            y_pred = test_pred
            y_target = torch.round(test_target)

            yt = y_target.detach().cpu().numpy()



            k = k + 1
            precision_score = precision_score + dice_loss.get_precision(y_pred,y_target)
            accuracy_score = accuracy_score + dice_loss.get_accuracy(y_pred,y_target)
            f1_score = f1_score + dice_loss.get_f1_score(y_pred,y_target)
            recall_score = recall_score + dice_loss.get_sensitivity(y_pred,y_target)
            MIoU = MIoU + dice_loss.get_MIoU(y_pred,y_target)

    print("precision_score:")
    print(precision_score / k)
    print('\n')
    print("accuracy_score:")
    print(accuracy_score / k)
    print('\n')
    print("f1_score:")
    print(f1_score / k)
    print('\n')
    print("recall_score:")
    print(recall_score / k)
    print('\n')
    print("MIoU:")
    print(MIoU[1] / k)
    print('\n')
    print("Number of parameter: %.2fM" % (total/1e6))
    print('\n')
    print("TestTime:")
    print(TestTime)
    print('\n')

if __name__ == '__main__':
    start = time.perf_counter()
    test()
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))

