# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:38:26 2023

@author: RYU
"""

import os
import time
import numpy as np
from numpy import argmax

import torch
from datetime import timedelta

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

#%%
def model_performance(labels, preds):
    
    """
    데이터의 실제 라벨값과 모델의 예측값을 받아 파라메터를 출력.
    2가지 class의 경우(binary classification)에만 이 함수를 사용할 수 있음.
    
    labels  : array, 데이터의 실제 라벨 값.
    preds   : array, 데이터에 대한 모델의 출력 값.
    
    
    <사람 기준>
    Sensitivity(민감도) : 실제 양성 환자들중에 양성으로 예측된 비율.   TP/(TP+FN)
    Specificity(특이도) : 실제 음성 환자들중에 음성으로 예측된 비율.   TN/(FP+TN)
    
    <측정기기, 모델 기준>
    PPV(Positive predictive value): 양성으로 예측한것중 실제로 양성이 있을 확률  TP/(TP+FP)
    NPV(Negative predictive vlaue): 음성으로 예측한것중 실제로 음성이 있을 확률  TN/(FN+TN)
    
    F1 score: precision(PPV)와 recall(sensitivity)의 가중조화평균
    
    """
    tn, fp, fn, tp = confusion_matrix(labels,preds).ravel()
    
    sen =  (tp/(tp+fn))*100
    spe =  (tn/(fp+tn))*100
    
    ppv =  (tp/(tp+fp))*100
    npv =  (tn/(fn+tn))*100
    
    f1  = ((2*ppv*sen)/(ppv+sen))
    
    return tn, fp, fn, tp, sen, spe, ppv, npv, f1

#%%

def Model_Train(args):
    
    train_loader      = args.train_loader
    validation_loader = args.validation_loader
    
    model     = args.model
    optimizer = args.optimizer
    criterion = args.criterion
    device = args.device
    
    
    
    """Train"""
    train_losses        = []
    avg_train_losses    = []
    Train_baths_ACC     = [] # batch 마다 train acc를 저장.
    Train_ACC           = [] # batch acc 평균.
    Train_AUROC         = []


    """Validaion"""
    valid_losses        = []
    avg_valid_losses    = []
    Validation_ACC      = []
    Valid_ACC_per_Class = []
    Validation_AUROC    = []
    
    
    """save best model"""
    best_acc = 0
    best_epoch = 0
    best_auroc = 0
    train_acc_cul = 0

    best_model_save_path = args.fold_path +'/'+ 'best model'
    
    start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        
        """Train"""
        model.train()
        train_loss = 0
        Train_baths_ACC = []

        true_labels  = np.array([]) # ideal label
        target_score = np.array([]) # ouput score
        prob_score   = np.array([]) # ouput prob
        
        for batch_idx, (data,target) in enumerate(train_loader):
    
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        

            optimizer.zero_grad()
            output = model(data).squeeze(dim=-1)
            pred = torch.round(torch.sigmoid(output))
            loss = criterion(output.view_as(target), target)
            
            """update and save loss"""
            train_loss += loss.item()
            loss.backward() # 가중치 갱신.
            optimizer.step()
            
            
            """calculate ACC"""
            correct = 0
            total = target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / total
            Train_baths_ACC.append(accuracy)
            
            
            """auroc inputs"""
            true_labels  = np.append(true_labels,target.detach().cpu().numpy())
            target_score = np.append(target_score,output.detach().cpu().numpy())
            prob_score   = np.append(prob_score,torch.sigmoid(output).detach().cpu().numpy())
            
            if batch_idx % args.log_interval == 0:
                #1.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

                #2.
                print('Train set:  batch loss: {:.4f}, Accuracy: {:.0f}% '.format(
                    loss.item() ,accuracy))
        
        train_auroc = roc_auc_score(true_labels, prob_score)
        Train_AUROC.append(train_auroc)
        
        
        """Validation"""
        model.eval()
        
        valid_loss = 0
        correct = 0
        total = len(validation_loader.dataset)

        true_labels  = np.array([]) # ideal label
        target_score = np.array([]) # ouput score
        prob_score   = np.array([]) # ouput prob
        pred_labels   = np.array([]) # model prediction
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

                output = model(data).squeeze(dim=-1)
                pred = torch.round(torch.sigmoid(output))

                """loss"""
                loss = criterion(output.view_as(target), target)
                valid_loss += loss.item()
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                
                """auroc inputs"""
                true_labels  = np.append(true_labels,np.array(target.cpu()))
                target_score = np.append(target_score,np.array(output.cpu()))
                prob_score   = np.append(prob_score,np.array(torch.sigmoid(output).cpu()))
                pred_labels  = np.append(pred_labels,np.array(torch.round(torch.sigmoid(output)).cpu()))
                
                
            valid_auroc = roc_auc_score(true_labels, prob_score)
            Validation_AUROC.append(valid_auroc)
            
            
        """Loss and ACC """
        # epoch당 평균 loss 계산
        train_loss /= len(train_loader)
        valid_loss /= len(validation_loader)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # epoch당 acc 계산
        Train_ACC.append(sum(Train_baths_ACC)/len(Train_baths_ACC))
        valid_accuracy = 100. * correct / total
        Validation_ACC.append(valid_accuracy)

        print('------------------------------------------')
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(valid_loss, correct, total, valid_accuracy))
        print('Valid set: AUROC: {:.4f}'.format(valid_auroc))
        print('-------------------------------------------')
        
        
        """Save best model"""
    
        if valid_auroc > best_auroc:
            torch.save(model, best_model_save_path)
            print("모델이 저장되었습니다.")
            print('-------------------------------------------')
            best_acc = valid_accuracy
            best_epoch = epoch
            best_auroc = valid_auroc
            
        print('----------------------------------------------------------------')
       
        
    end = time.time()
    print('----------------------------------------------------------------')
    print('[System Complete: {}]'.format(timedelta(seconds=end-start)))
    
    
    
    
    return best_acc, best_auroc, best_epoch, avg_train_losses, avg_valid_losses, Train_ACC, Validation_ACC, Train_AUROC, Validation_AUROC
      
#%%

def Model_Test(args):
    
    test_loader  = args.test_loader
    
    device = args.device
    criterion = args.criterion
    
    best_model_save_path = args.fold_path +'/'+ 'best model'
    model = torch.load(best_model_save_path)
    model.to(device)  
    
    true_labels  = np.array([]) # 실제 라벨 값.
    target_score = np.array([]) 
    prob_score   = np.array([]) # ouput prob
    pred_labels  = np.array([]) # 모델의 예측값.(0혹은1로 변환.)
    
    
    if args.test_batch_size ==1:
        print("test batch size is 1")
        pass
    else:
        return print("test batch size is not 1")
    
    
    """test"""
    model.eval()

    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            output = model(data).squeeze(dim=-1)
            pred = torch.round(torch.sigmoid(output))
            
            
            """loss"""
            loss = criterion(output.view_as(target), target)
            test_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()


            """결과값 누적."""
            true_labels = np.append(true_labels,np.array(target.cpu())) # 실제 라벨
            target_score = np.append(target_score,np.array(output.cpu()))
            prob_score   = np.append(prob_score,np.array(torch.sigmoid(output).cpu()))
            pred_labels = np.append(pred_labels,np.array(pred.cpu()))  # 예측 결과

            print('test: [{}/{}] '.format(batch_idx,len(test_loader)-1))
            
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total, test_accuracy))
    
    
    """폴더 만들기"""
    test_path = args.fold_path +'/test'

    try:
        if not(os.path.isdir(test_path)):
            os.makedirs(os.path.join(test_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
            
    """test시 라벨값, pred값"""
    np.save(test_path +'/true_labels.npy',true_labels)
    np.save(test_path +'/target_score.npy',target_score)
    np.save(test_path +'/prob_score.npy',prob_score)
    
    
    
    tn, fp, fn, tp, sen,spe,ppv,npv,f1 = model_performance(true_labels,pred_labels)
    
    test_dict = {}
    test_dict['tn'] = tn
    test_dict['fp'] = fp
    test_dict['fn'] = fn
    test_dict['tp'] = tp
    test_dict['sen'] = sen
    test_dict['spe'] = spe
    test_dict['ppv'] = ppv
    test_dict['npv'] = npv
    test_dict['f1'] = f1
    test_dict['ACC'] = test_accuracy
    test_dict['AUROC'] = roc_auc_score(true_labels, prob_score)
    
    
    return test_dict

#%%
def Model_Test_Youden_Index(args):
    
    validation_loader = args.validation_loader
    test_loader  = args.test_loader
    
    device = args.device
    criterion = args.criterion
    
    best_model_save_path = args.fold_path +'/'+ 'best model'
    model = torch.load(best_model_save_path)
    model.to(device)  
    
    if args.test_batch_size ==1:
        print("test batch size is 1")
        pass
    else:
        return print("test batch size is not 1")
    
    
    """get Youden Index"""
    true_labels  = np.array([]) # 실제 라벨 값.
    target_score = np.array([]) 
    prob_score   = np.array([]) # ouput prob
    pred_labels  = np.array([]) # 모델의 예측값.(0혹은1로 변환.)
    
    model.eval()

    test_loss = 0
    correct = 0
    total = len(validation_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            
            
            output = model(data).squeeze(dim=-1)

            """pred(BCE loss)"""
            pred = torch.round(torch.sigmoid(output))

            """결과값 누적."""
            true_labels = np.append(true_labels,np.array(target.cpu())) # 실제 라벨
            target_score = np.append(target_score,np.array(output.cpu()))
            prob_score   = np.append(prob_score,np.array(torch.sigmoid(output).cpu()))
            pred_labels = np.append(pred_labels,np.array(pred.cpu()))  # 예측 결과

            print('validation: [{}/{}] '.format(batch_idx,len(validation_loader)-1))
            

            
            
    """Youden’s J statistic. / J = Sensitivity + Specificity – 1"""

    # calculate roc curves
    FPR, TPR, thresholds = roc_curve(true_labels, prob_score)

    # get the best threshold
    J = TPR - FPR
    idx = argmax(J)
    best_thresh = thresholds[idx]

    print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, TPR[idx], 1-FPR[idx], J[idx]))
            
    
    
    
    """test with youden index"""
    true_labels  = np.array([]) # 실제 라벨 값.
    target_score = np.array([]) # 모델의 출력값.
    prob_score   = np.array([]) # ouput prob
    pred_labels  = np.array([]) # 모델의 예측값.(0혹은1로 변환.)
    
    """test"""
    model.eval()

    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            

            output = model(data).squeeze(dim=-1)
            pred = (torch.sigmoid(output) > best_thresh).int() 


            loss = criterion(output.view_as(target), target)
            test_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()


            """결과값 누적."""
            true_labels = np.append(true_labels,np.array(target.cpu())) # 실제 라벨
            target_score = np.append(target_score,np.array(output.cpu()))
            prob_score   = np.append(prob_score,np.array(torch.sigmoid(output).cpu()))
            pred_labels = np.append(pred_labels,np.array(pred.cpu()))  # 예측 결과

            print('test: [{}/{}] '.format(batch_idx,len(test_loader)-1))


    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total, test_accuracy))
    
    
    """폴더 만들기"""
    test_path = args.fold_path +'/test_youden'

    try:
        if not(os.path.isdir(test_path)):
            os.makedirs(os.path.join(test_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
            
    """test시 라벨값, pred값"""
    np.save(test_path +'/true_labels.npy',true_labels)
    np.save(test_path +'/target_score.npy',target_score)
    np.save(test_path +'/prob_score.npy',prob_score)
    
    
    
    tn, fp, fn, tp, sen,spe,ppv,npv,f1 = model_performance(true_labels,pred_labels)
    
    test_dict = {}
    test_dict['idx'] = idx
    test_dict['best threshold'] = best_thresh
    test_dict['sensitivity_in_best threshold'] =  TPR[idx]
    test_dict['specificity_in_best threshold'] =  1-FPR[idx]
    test_dict['J_best threshold'] =  J[idx]
    
    test_dict['tn'] = tn
    test_dict['fp'] = fp
    test_dict['fn'] = fn
    test_dict['tp'] = tp
    test_dict['sen'] = sen
    test_dict['spe'] = spe
    test_dict['ppv'] = ppv
    test_dict['npv'] = npv
    test_dict['f1'] = f1
    test_dict['ACC'] = test_accuracy
    test_dict['AUROC'] = roc_auc_score(true_labels, prob_score)
    
    
    return test_dict

#%%

def T_SNE(args):
    
    """
    path: 그림 저장할 위치.
    """
    
    train_loader = args.train_loader
    test_loader  = args.test_loader
    
    device = args.device
    
    best_model_save_path = args.fold_path +'/'+ 'best model of fold ' + str(args.Fold)
    model = torch.load(best_model_save_path)
    model.to(device) 
    
    """feature 및 라벨 축적"""
    
    model.eval()
    
    potential_features = np.array([])
    labels_train       = np.array([])
    labels_test        = np.array([])
    
    with torch.no_grad():
        for batch_idx, (data,_,target) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float)
            output = model.get_feature(data)
            
            if batch_idx == 0:
                potential_features = output.cpu()
            else:
                potential_features = np.vstack((potential_features,output.cpu()))
            labels_train = np.append(labels_train, np.array(target.cpu()))
    
            #print('[{}/{}] '.format(batch_idx,len(train_loader)-1))
            
            
    with torch.no_grad():
        for batch_idx, (data,_,target) in enumerate(test_loader):
            data = data.to(device, dtype=torch.float)
            output = model.get_feature(data)
            
            potential_features = np.vstack((potential_features,output.cpu()))
            labels_test = np.append(labels_test, np.array(target.cpu()))

            #print('[{}/{}] '.format(batch_idx,len(test_loader)-1))
            
    """T SNE"""
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10, n_iter=1000)
    embedded_features = tsne.fit_transform(potential_features)
    
    
    train_normal   = []
    train_abnormal = []
    test_normal    = []
    test_abnormal  = []
    
    for i in range(len(labels_train)+len(labels_test)):
        
        
        if i < len(labels_train):
            
            tempt = str(int(labels_train[i])) # 라벨
            
            if tempt == '0':
                train_normal.append(embedded_features[i])
            else:
                train_abnormal.append(embedded_features[i])
                
        elif i >= len(labels_train):
            
            tempt = str(int(labels_test[i-len(labels_train)]))
    
            if tempt == '0':
                test_normal.append(embedded_features[i])
            else:
                test_abnormal.append(embedded_features[i])
    
    """Figure"""
    color_list =["#D93240", "#2621BF"]
    label_list = [ r"normal(train)", r"abnormal(train)",  r"normal(train)", r"abnormal(test)"]
    marker_list = ['o', 'x']
    

    """Cobbined"""
    fig=plt.figure(figsize=(15, 15))

    plt.scatter(x=np.array(train_normal)[:,0], y=np.array(train_normal)[:,1], 
                color= color_list[0], alpha=0.7, marker=marker_list[0], label=label_list[0])
    plt.scatter(x=np.array(train_abnormal)[:,0], y=np.array(train_abnormal)[:,1], 
                color= color_list[1], alpha=0.7, marker=marker_list[0], label=label_list[1])
    
    plt.scatter(x=np.array(test_normal)[:,0], y=np.array(test_normal)[:,1], 
                color= color_list[0], alpha=0.7, marker=marker_list[1], label=label_list[2])
    plt.scatter(x=np.array(test_abnormal)[:,0], y=np.array(test_abnormal)[:,1], 
                color= color_list[1], alpha=0.7, marker=marker_list[1], label=label_list[3])
    
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.xticks(range(-40,50,10), fontsize = 15)
    plt.yticks(range(-40,50,10), fontsize = 15)
    
    plt.grid(True)
    plt.legend(loc='lower right', fontsize = 15)
    plt.tight_layout()
    
    #plt.show()
    
    plt.savefig(args.fold_path+'/'+'T-SNE Combined', dpi = 200) 
    
    plt.cla()
    plt.clf()
    plt.close()
    

    """Train"""
    fig=plt.figure(figsize=(15, 15))

    plt.scatter(x=np.array(train_normal)[:,0], y=np.array(train_normal)[:,1], 
                color= color_list[0], alpha=0.7, marker=marker_list[0], label=label_list[0])
    plt.scatter(x=np.array(train_abnormal)[:,0], y=np.array(train_abnormal)[:,1], 
                color= color_list[1], alpha=0.7, marker=marker_list[0], label=label_list[1])
    
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.xticks(range(-40,50,10), fontsize = 15)
    plt.yticks(range(-40,50,10), fontsize = 15)
    
    plt.grid(True)
    plt.legend(loc='lower right', fontsize = 15)
    plt.tight_layout()
    
    #plt.show()
    
    plt.savefig(args.fold_path+'/'+'T-SNE Train', dpi = 200)
    
    plt.cla()
    plt.clf()
    plt.close()
    
    
    """Test"""
    fig=plt.figure(figsize=(15, 15))

    
    plt.scatter(x=np.array(test_normal)[:,0], y=np.array(test_normal)[:,1], 
                color= color_list[0], alpha=0.7, marker=marker_list[1], label=label_list[2])
    plt.scatter(x=np.array(test_abnormal)[:,0], y=np.array(test_abnormal)[:,1], 
                color= color_list[1], alpha=0.7, marker=marker_list[1], label=label_list[3])
    
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.xticks(range(-40,50,10), fontsize = 15)
    plt.yticks(range(-40,50,10), fontsize = 15)
    
    plt.grid(True)
    plt.legend(loc='lower right', fontsize = 15)
    plt.tight_layout()
    
    #plt.show()
    
    plt.savefig(args.fold_path+'/'+'T-SNE Test', dpi = 200) 
        
    plt.cla()
    plt.clf()
    plt.close()