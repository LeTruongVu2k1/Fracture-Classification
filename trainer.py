import torch.backends.cudnn as cudnn
from PIL import Image
import torch
from tqdm import tqdm
import time
from barbar import Bar
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
import numpy as np
from collections import defaultdict
import os
from torch.cuda.amp import GradScaler, autocast
from asymmetric_loss import AsymmetricLoss, AsymmetricLossOptimized
import pickle 
from collections import defaultdict
import torch.nn as nn
import math

use_gpu = True

# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets

# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


class CheXpertTrainer_Asymmetric():

    def train(args, model, dataLoaderTrain, dataLoaderVal, nnClassCount):
        trMaxEpoch = args.epoch
        checkpoint = args.checkpoint
        mode = args.mode
        device = args.device
        model_path = args.path
        weighted_class = args.weighted_score
        p_mixup = args.p_mixup
        is_triple = args.is_triple

        os.makedirs(f'{model_path}/{mode}/logs/', exist_ok=True) 
        log_dict = defaultdict(list) 
        
        lr = 0.0001
        weight_decay = 0.001
        print(f'Learning rate: {lr}')
        print(f'Weight Decay: {weight_decay}')
        print('*************** Start Training ***************')
        
#         optimizer = torch.optim.Adam(model.parameters(), lr = lr, # setting optimizer & scheduler
#                                betas = (0.9, 0.999), eps = 1e-08, weight_decay = weight_decay) 
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, # setting optimizer & scheduler
                               betas = (0.9, 0.999), eps = 1e-08, weight_decay = weight_decay) 
        
        log_dict['lr'].append(lr)
        log_dict['weight_decay'].append(weight_decay)
        
        asym_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False)
        scaler = GradScaler()
        start_epoch = 0
        BEST_VAL_F1 = -1
        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
#             scaler.load_state_dict(modelCheckpoint['scaler'])
            start_epoch = modelCheckpoint['epoch']
            BEST_VAL_F1 = modelCheckpoint['best_f1']
            
        # Train the network
        records = {}
        
        train_start = []
        train_end = []
        for epochID in range(start_epoch, start_epoch + trMaxEpoch):
            start_time = time.time()
            
            train_start.append(time.time()) # training starts

            if is_triple:
                losst, best_f1_t, best_threshold_t = CheXpertTrainer_Asymmetric.epochTrain_Triple(model, dataLoaderTrain, optimizer, scaler, trMaxEpoch, nnClassCount, device, asym_criterion, p_mixup)
            else:
                losst, best_f1_t, best_threshold_t = CheXpertTrainer_Asymmetric.epochTrain(args, model, dataLoaderTrain, optimizer, scaler, trMaxEpoch, nnClassCount, device, asym_criterion, p_mixup)

            train_end.append(time.time()) # training ends
            lossv, best_f1_v, best_threshold_v = CheXpertTrainer_Asymmetric.epochVal(args, model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, device, asym_criterion)
            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
            print(f"Training F1: {best_f1_t} - Threshold: {best_threshold_t}")
            
            if best_f1_v > BEST_VAL_F1 or (epochID == start_epoch + trMaxEpoch - 1):                                

                os.makedirs(f'{model_path}/{mode}', exist_ok=True) 
                if epochID==(start_epoch + trMaxEpoch - 1) and best_f1_v <= BEST_VAL_F1:                   
                    save_path = f'{model_path}/{mode}/last_epoch_{epochID+1}' + '.pth.tar' 
                else:
                    save_path = f'{model_path}/{mode}/epoch_{epochID+1}' + '.pth.tar' 
                    
                BEST_VAL_F1 = max(best_f1_v, BEST_VAL_F1)    
                
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(),
                            'best_f1': BEST_VAL_F1, 'optimizer': optimizer.state_dict()}, #, 'scaler': scaler.state_dict()},
                           save_path)
                
                print('Epoch ' + str(epochID + 1) + ' [save] F1-score on Validation Set = ' + str(BEST_VAL_F1) + f' - Threshold: {best_threshold_v}')
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] F1-score on Validation Set = ' + str(best_f1_v) + f' - Threshold: {best_threshold_v}')
            
            end_time = time.time()
            
            print(f'Training & evaluating on validation take: {(end_time - start_time)/60} minutes')
            print('------------------------------------------------------------')

            records['epoch'] = epochID
            records['train_loss'] = losst
            records['val_loss'] = lossv
            
            ########## Logging ########## 
            log_dict['Epoch'].append(epochID + 1)
            log_dict['Train Loss'].append(losst)
            log_dict['Valid Loss'].append(lossv)
            log_dict['Train F1'].append(best_f1_t)
            log_dict['Valid F1'].append(best_f1_v)
            log_dict['Train Threshold'].append(best_threshold_t)
            log_dict['Valid Threshold'].append(best_threshold_v)
            log_dict["Train+Evaluating's time"].append((end_time - start_time)/60)

        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))
        params = model.state_dict()
        
        os.makedirs(f'{model_path}/{mode}/logs/', exist_ok=True)            
        filename = f'{model_path}/{mode}/logs/{start_epoch}_{start_epoch + trMaxEpoch}.pickle'
        filehandler = open(filename, 'wb')
        pickle.dump(log_dict, filehandler)
        filehandler.close()

      
        return params, records


    def epochTrain(args, model, dataLoaderTrain, optimizer, scaler, epochMax, classCount, device, criterion, p_mixup):
        losstrain = 0
        model.train()
        outGT = torch.FloatTensor().cpu()
        outPRED = torch.FloatTensor().cpu()    
        
        outGT_paths = []
        outPRED_paths = []
        ROOT = f'{args.path}/{args.mode}/preds_grounds'
        os.makedirs(ROOT, exist_ok=True)    
        last_batch_id = math.ceil(len(dataLoaderTrain.dataset) / dataLoaderTrain.batch_size)
        saved_spot = round(30000 // dataLoaderTrain.batch_size, -2) # 30000 is experiment
        
        for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):                                   
            optimizer.zero_grad()
            
            varInput = varInput.to(device)
            varTarget = target.to(device)
            
            p = np.random.rand()
            if p < p_mixup:
                varInput, varTarget = mixup(varInput, varTarget, 0.8)
                
#             with torch.autocast(device_type='cuda', dtype=torch.float16):
            varOutput = model(varInput, mode='train')
    
            if p < p_mixup:
                lossvalue = mixup_criterion(varOutput, varTarget)
            else: 
                lossvalue = criterion(varOutput, varTarget)
            
#             scaler.scale(lossvalue).backward()
#             scaler.step(optimizer)
#             scaler.update()
            lossvalue.backward()
            optimizer.step()
            
            losstrain += lossvalue.item()
            
            outGT = torch.cat((outGT, target), 0)
            outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
            #### Reset variables `outGT` and `outPRED` to free CPU's memory (these variables are accumulated during training)
            if batchID % saved_spot == 0 or batchID == last_batch_id:
                torch.save(outGT, f'{ROOT}/outGT_{batchID}.pt')
                torch.save(outPRED, f'{ROOT}/outPRED_{batchID}.pt')
                outGT = torch.FloatTensor().cpu()
                outPRED = torch.FloatTensor().cpu()   
                outGT_paths.append(f'{ROOT}/outGT_{batchID}.pt')
                outPRED_paths.append(f'{ROOT}/outPRED_{batchID}.pt')
            
        outGT = torch.FloatTensor().cpu()
        outPRED = torch.FloatTensor().cpu()     
        for outGT_path, outPRED_path in zip(outGT_paths, outPRED_paths):
            outGT = torch.cat((outGT, torch.load(outGT_path)), 0)
            outPRED = torch.cat((outPRED, torch.load(outPRED_path)), 0)
                
        best_f1, best_threshold = CheXpertTrainer_Asymmetric.compute_f1_with_threshold(outGT, outPRED)

        return losstrain / len(dataLoaderTrain), best_f1, 1/(1 + np.exp(-best_threshold))  # sigmoid(best_threshold)

 
    def epochTrain_TripleLoss(model, dataLoaderTrain, optimizer, scaler, epochMax, classCount, device, criterion):
        losstrain = 0
        model.train()
        outGT = torch.FloatTensor().cpu()
        outPRED = torch.FloatTensor().cpu()     

        for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):                                   
            optimizer.zero_grad()

            varInput = varInput.to(device)
            varTarget = target.to(device)

            varOutput = model(varInput, mode='train')

            lossvalue = triple_criterion(varOuput, varTarget)

            lossvalue.backward()
            optimizer.step()

            losstrain += lossvalue.item()
            
            # Get prediction and groundtruth of concatenation between frontal-lateral
            outGT = torch.cat((outGT, target[:, -1]), 0)
            outPRED = torch.cat((outPRED, varOutput[:, -1].cpu()), 0)

        best_f1, best_threshold = CheXpertTrainer_Asymmetric.compute_f1_with_threshold(outGT, outPRED)

        return losstrain / len(dataLoaderTrain), best_f1, 1/(1 + np.exp(-best_threshold))  # sigmoid(best_threshold)


    def epochVal(args, model, dataLoaderVal, optimizer, epochMax, classCount, device, criterion):
        model.eval()
        lossVal = 0
        outGT = torch.FloatTensor().cpu()
        outPRED = torch.FloatTensor().cpu()   
        
        outGT_paths = []
        outPRED_paths = []
        VAL_ROOT = f'{args.path}/{args.mode}/preds_grounds_validation'
        os.makedirs(VAL_ROOT, exist_ok=True)            
        last_batch_id = math.ceil(len(dataLoaderVal.dataset) / dataLoaderVal.batch_size)
        saved_spot = round(30000 // dataLoaderVal.batch_size, -2) # 30000 is experiment

        with torch.no_grad():
            for i, (varInput, target) in enumerate(Bar(dataLoaderVal)):    
                varInput = varInput.to(device)
                target = target.to(device)
                varOutput = model(varInput, mode='val')
            
                lossVal += criterion(varOutput, target).item()
                
                outGT = torch.cat((outGT, target.cpu()), 0)
                outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
                if i % saved_spot == 0 or i == last_batch_id:
                    torch.save(outGT, f'{VAL_ROOT}/outGT_{i}.pt')
                    torch.save(outPRED, f'{VAL_ROOT}/outPRED_{i}.pt')
                    outGT = torch.FloatTensor().cpu()
                    outPRED = torch.FloatTensor().cpu()   
                    outGT_paths.append(f'{VAL_ROOT}/outGT_{i}.pt')
                    outPRED_paths.append(f'{VAL_ROOT}/outPRED_{i}.pt')
                
        best_f1, best_threshold = CheXpertTrainer_Asymmetric.compute_f1_with_threshold(outGT, outPRED)

        return lossVal / len(dataLoaderVal), best_f1, best_threshold 


    def compute_precision_recall_fscore(dataGT, dataPRED, classCount):
        # Computes area under ROC curve
        # dataGT: ground truth data
        # dataPRED: predicted data
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        scores = defaultdict(list)
        averages = ['binary', 'micro', 'macro', 'weighted']
        for a in averages:
          precision, recall, f_1, support = precision_recall_fscore_support(datanpGT, datanpPRED, average=a)
          scores['precision'].append(precision)
          scores['recall'].append(recall)
          scores['f_1'].append(f_1)
        return scores

    def computeAUROC(dataGT, dataPRED, classCount):
        # Computes area under ROC curve
        # dataGT: ground truth data
        # dataPRED: predicted data
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    def compute_f1_with_threshold(dataGT, dataPRED):
        datanpGT = dataGT.cpu().detach().numpy()
        datanpPRED = dataPRED.cpu().detach().numpy()
        
        precision, recall, threshold = precision_recall_curve(datanpGT, datanpPRED, pos_label=1, drop_intermediate=True)
        
        f1_scores = 2*(precision*recall)/(precision+recall)
        f1_scores = np.nan_to_num(f1_scores) # replace np.nan value to 0
        best_f1 = np.max(f1_scores)
        best_threshold = threshold[np.argmax(f1_scores)]
        
        return best_f1, best_threshold

    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, device):
        cudnn.benchmark = True

        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        model.eval()
        
        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(dataLoaderTest)):
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).to(device)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w).to(device)

                out = model(varInput, mode='test')
                outPRED = torch.cat((outPRED, out), 0)
                
                end = time.time()

        threshold = torch.Tensor([0.5]).to(device)
        outPRED = (outPRED > threshold).float() * 1
#         outPRED = torch.round(outPRED)

        aurocIndividual = CheXpertTrainer_Asymmetric.computeAUROC(outGT, outPRED, nnClassCount)
#         outGT = outGT.cpu().numpy()
#         outPRED = outPRED.cpu().numpy()
#         for i in range(nnClassCount):
#             print("*"*20)
#             print(confusion_matrix(outGT[:, i], outPRED[:, i]))

        aurocMean = np.array(aurocIndividual).mean()
        print('AUROC mean ', aurocMean)

        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])

        scores = CheXpertTrainer_Asymmetric.compute_precision_recall_fscore(outGT, outPRED, nnClassCount)

        print(f"precision: {scores['precision']} - recall: {scores['recall']} - f1: {scores['f_1']}")

        return outGT, outPRED

    def triple_criterion(outputs, targets):
        front_output, lat_output, both_output = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        front_target, lat_target, both_target = targets[:, 0], targets[:, 1], targets[:, 2]

        front_criterion, lat_criterion, both_criterion = criterion(front_output, front_target), criterion(lat_output, lat_target), criterion(both_output, both_target)

        return 1/2*front_criterion + 1/2*lat_criterion + both_criterion

 
#############################################################################################
    
class CheXpertTrainer():
    pass
#     def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint, mode, device, model_path, weighted_class):
#         log_dict = defaultdict(list)
        
#         lr = 0.001
#         weight_decay = 0.01
#         optimizer = torch.optim.Adam(model.parameters(), lr = lr, # setting optimizer & scheduler
#                                betas = (0.9, 0.999), eps = 1e-08, weight_decay = weight_decay)        
        
#         log_dict['lr']  = lr
#         log_dict['weight_decay'] = weight_decay
        
#         scaler = GradScaler()
# #         loss = torch.nn.BCELoss() # setting loss function
#         start_epoch = 0
#         BEST_VAL_F1 = -1
#         if checkpoint != None and use_gpu: # loading checkpoint
#             modelCheckpoint = torch.load(checkpoint)
#             model.load_state_dict(modelCheckpoint['state_dict'])
#             optimizer.load_state_dict(modelCheckpoint['optimizer'])
# #             scaler.load_state_dict(modelCheckpoint['scaler'])
#             start_epoch = modelCheckpoint['epoch']
#             BEST_VAL_F1 = modelCheckpoint['best_f1']
            
#         # Train the network
#         records = {}
        
#         train_start = []
#         train_end = []
#         for epochID in range(start_epoch, start_epoch + trMaxEpoch):
#             start_time = time.time()
            
#             train_start.append(time.time()) # training starts
#             losst, best_f1_t, best_threshold_t = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, scaler, trMaxEpoch, nnClassCount, device, weighted_class)
#             train_end.append(time.time()) # training ends
#             lossv, best_f1_v, best_threshold_v = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, device, weighted_class)
#             print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
#             print(f"Training F1: {best_f1_t} - Threshold: {best_threshold_t}")
            
#             if best_f1_v > BEST_VAL_F1:
#                 os.makedirs(f'{model_path}/{mode}', exist_ok = True) 
                
#                 BEST_VAL_F1 = max(best_f1_v, BEST_VAL_F1)
#                 torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(),
#                             'best_f1': BEST_VAL_F1, 'optimizer' : optimizer.state_dict(), 'scaler': scaler.state_dict()},
#                            f'{model_path}/{mode}/epoch_{epochID+1}' + '.pth.tar')
#                 print('Epoch ' + str(epochID + 1) + ' [save] F1-score on Validation Set = ' + str(BEST_VAL_F1) + f' - Threshold: {best_threshold_v}')
#             else:
#                 print('Epoch ' + str(epochID + 1) + ' [----] F1-score on Validation Set = ' + str(best_f1_v) + f' - Threshold: {best_threshold_v}')
            
#             end_time = time.time()
            
#             print(f'Training & evaluating on validation take: {(end_time - start_time)/60} minutes')
#             print('------------------------------------------------------------')
#             records['epoch'] = epochID
#             records['train_loss'] = losst
#             records['val_loss'] = lossv
            
#             ########## Logging ########## 
#             log_dict['Epoch'].append(epochID + 1)
#             log_dict['Train Loss'].append(losst)
#             log_dict['Valid Loss'].append(lossv)
#             log_dict['Train F1'].append(best_f1_t)
#             log_dict['Valid F1'].append(best_f1_v)
#             log_dict['Train Threshold'].append(best_threshold_t)
#             log_dict['Valid Threshold'].append(best_threshold_v)
#             log_dict["Train+Evaluating's time"].append((end_time - start_time)/60)

#         train_time = (np.array(train_end) - np.array(train_start)) / 60
#         print("Training time for each epoch: {} seconds".format(train_time.round(0)))
#         params = model.state_dict()
        

#         ########## Write Log to file ##########
#         filename = f'./logs/{start_epoch}_{start_epoch + trMaxEpoch}.pickle'
#         filehandler = open(filename, 'wb')
#         pickle.dump(log_dict, filehandler)
#         filehandler.close()
        
#         return params, records


#     def epochTrain(model, dataLoaderTrain, optimizer, scaler, epochMax, classCount, device, weighted_class):
#         losstrain = 0
#         model.train()
#         outGT = torch.FloatTensor().cpu()
#         outPRED = torch.FloatTensor().cpu()     
        
#         for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):            
#             weights = None
#             if weighted_class:
#                 weights = torch.where(target == 0.0, weighted_class['neg_label'], weighted_class['pos_label']).to(device)
#             loss = torch.nn.BCEWithLogitsLoss(weight=weights)
            
#             optimizer.zero_grad()
            
#             varInput = varInput.to(device)
#             varTarget = target.to(device)
            
#             with torch.autocast(device_type='cuda', dtype=torch.float16):
#                 varOutput = model(varInput, mode='train')
#                 lossvalue = loss(varOutput, varTarget)

#             scaler.scale(lossvalue).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             losstrain += lossvalue.item()
            
#             outGT = torch.cat((outGT, varTarget.cpu()), 0)
#             outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
                            
#         best_f1, best_threshold = CheXpertTrainer.compute_f1_with_threshold(outGT, outPRED)

#         return losstrain / len(dataLoaderTrain), best_f1, 1/(1 + np.exp(-best_threshold))  # sigmoid(best_threshold)


#     def epochVal(model, dataLoaderVal, optimizer, epochMax, classCount, device, weighted_class):
#         model.eval()
#         lossVal = 0
#         outGT = torch.FloatTensor().cpu()
#         outPRED = torch.FloatTensor().cpu()     
        
#         with torch.no_grad():
#             for i, (varInput, target) in enumerate(Bar(dataLoaderVal)):    
#                 varInput = varInput.to(device)
#                 target = target.to(device)
#                 varOutput = model(varInput, mode='val')
                
#                 weights = None
#                 if weighted_class:
#                     weights = torch.where(target == 0.0, weighted_class['neg_label'], weighted_class['pos_label']).to(device)

#                 loss = torch.nn.BCELoss(weight=weights)
            
#                 lossVal += loss(varOutput, target)
                
#                 outGT = torch.cat((outGT, target.cpu()), 0)
#                 outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
                
#         best_f1, best_threshold = CheXpertTrainer.compute_f1_with_threshold(outGT, outPRED)

#         return lossVal / len(dataLoaderVal), best_f1, best_threshold


#     def compute_precision_recall_fscore(dataGT, dataPRED, classCount):
#         # Computes area under ROC curve
#         # dataGT: ground truth data
#         # dataPRED: predicted data
#         datanpGT = dataGT.cpu().numpy()
#         datanpPRED = dataPRED.cpu().numpy()

#         scores = defaultdict(list)
#         averages = ['binary', 'micro', 'macro', 'weighted']
#         for a in averages:
#           precision, recall, f_1, support = precision_recall_fscore_support(datanpGT, datanpPRED, average=a)
#           scores['precision'].append(precision)
#           scores['recall'].append(recall)
#           scores['f_1'].append(f_1)
#         return scores

#     def computeAUROC(dataGT, dataPRED, classCount):
#         # Computes area under ROC curve
#         # dataGT: ground truth data
#         # dataPRED: predicted data
#         outAUROC = []
#         datanpGT = dataGT.cpu().numpy()
#         datanpPRED = dataPRED.cpu().numpy()

#         for i in range(classCount):
#             try:
#                 outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
#             except ValueError:
#                 pass
#         return outAUROC
    
#     def compute_f1_with_threshold(dataGT, dataPRED):
#         datanpGT = dataGT.cpu().detach().numpy()
#         datanpPRED = dataPRED.cpu().detach().numpy()
        
#         precision, recall, threshold = precision_recall_curve(datanpGT, datanpPRED, pos_label=1, drop_intermediate=True)
        
#         f1_scores = 2*(precision*recall)/(precision+recall)
#         f1_scores = np.nan_to_num(f1_scores) # replace np.nan value to 0
#         best_f1 = np.max(f1_scores)
#         best_threshold = threshold[np.argmax(f1_scores)]
        
#         return best_f1, best_threshold

#     def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, device):
#         cudnn.benchmark = True

#         if checkpoint != None and use_gpu:
#             modelCheckpoint = torch.load(checkpoint)
#             model.load_state_dict(modelCheckpoint['state_dict'])

#         if use_gpu:
#             outGT = torch.FloatTensor().cuda()
#             outPRED = torch.FloatTensor().cuda()
#         else:
#             outGT = torch.FloatTensor()
#             outPRED = torch.FloatTensor()

#         model.eval()
        
#         end = time.time()
#         with torch.no_grad():
#             for i, (input, target) in enumerate(tqdm(dataLoaderTest)):
#                 target = target.cuda()
#                 outGT = torch.cat((outGT, target), 0).to(device)

#                 bs, c, h, w = input.size()
#                 varInput = input.view(-1, c, h, w).to(device)

#                 out = model(varInput, mode='test')
#                 outPRED = torch.cat((outPRED, out), 0)
                
#                 end = time.time()

#         threshold = torch.Tensor([0.5]).to(device)
#         outPRED = (outPRED > threshold).float() * 1
# #         outPRED = torch.round(outPRED)

#         aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
# #         outGT = outGT.cpu().numpy()
# #         outPRED = outPRED.cpu().numpy()
# #         for i in range(nnClassCount):
# #             print("*"*20)
# #             print(confusion_matrix(outGT[:, i], outPRED[:, i]))

#         aurocMean = np.array(aurocIndividual).mean()
#         print('AUROC mean ', aurocMean)

#         for i in range (0, len(aurocIndividual)):
#             print(class_names[i], ' ', aurocIndividual[i])

#         scores = CheXpertTrainer.compute_precision_recall_fscore(outGT, outPRED, nnClassCount)

#         print(f"precision: {scores['precision']} - recall: {scores['recall']} - f1: {scores['f_1']}")

#         return outGT, outPRED