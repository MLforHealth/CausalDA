import numpy as np
import logging
import torch
import torch.nn.functional as F
from sklearn import metrics 
import pdb

## Get the same logger from main"
logger = logging.getLogger("causal_bootstrap")

def cost_fn(out, tar, weighting):
    return F.cross_entropy(out, tar, weight=weighting, size_average=False).item()

def prediction_analysis(args, model, device, data_loader, batch_size):
    
    # logger.info("Starting Evaluation")
    model.eval() # not training model 

    total_loss = 0 ; total_acc  = 0 
    prob_list = []; pred_list = [];  target_list = []  

    with torch.no_grad():
        for [data, target, extras] in data_loader:
            data = data.float().to(device) # add channel dimension
            target = target.long().to(device)
            if torch.is_tensor(extras):
                extras = extras.float().to(device)

            if args.data == "MNIST_deep":
                data = data.reshape(data.shape[0],-1)

            output = model(data, extras)
            target = target.view((-1,))            

            target_list = target_list + list(target.cpu().detach().tolist())

            if args.data == "NIH" or args.data == "MIMIC" : 
                weights = torch.tensor([0.1,0.9], dtype = torch.float).to(device)
            if args.data == "NIH_multi":
                weights = torch.tensor([0.1,0.8,0.1], dtype = torch.float).to(device)
            if args.data == "CelebA_attr": 
                weights = torch.tensor([0.2,0.8], dtype = torch.float).to(device)
            else:
                weights = None 
            
            total_loss += cost_fn(output, target, weighting=weights) # sum up batch loss
            
            prob = F.softmax(output, dim=1)
            prob_list = prob_list + list(prob.cpu().detach().tolist())
            
            pred = prob.max(1, keepdim=True)[1]
            pred_list = pred_list + list(pred.cpu().detach().tolist())
            
            # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset)# average loss
    total_acc  /= 1.*len(data_loader.dataset) # average acc

    # pdb.set_trace()

    if len(np.unique(target_list)) > 2: 
        target_list = np.array(target_list)
        tar_one = np.zeros((target_list.size, target_list.max()+1))
        tar_one[np.arange(target_list.size), target_list] = 1
        prob_list = np.array(prob_list)

    else: 
        target_list = np.squeeze(np.array(target_list))
        prob_list = np.squeeze(np.array(prob_list))[:,1]
        pred_list = np.squeeze(np.array(pred_list))

    if len(np.unique(target_list)) > 2: 
        AUC_wo = metrics.roc_auc_score(tar_one,prob_list,average='weighted',multi_class='ovo')
        AUC_mo = metrics.roc_auc_score(tar_one,prob_list,average='macro',multi_class='ovo')
        AUC_wr = metrics.roc_auc_score(tar_one,prob_list,average='weighted',multi_class='ovr')
        AUC_mr = metrics.roc_auc_score(tar_one,prob_list,average='macro',multi_class='ovr')
        logger.info('===> Validation set: Average loss: {:.4f}\tAUC_wo: {:.4f}\tAUC_mo: {:.4f}\tAUC_wr: {:.4f}\tAUC_mr: {:.4f}\n'.format(
                    total_loss, AUC_wo, AUC_mo, AUC_wr, AUC_mr))
        
        AUC = [AUC_wo, AUC_mo, AUC_wr, AUC_mr]

    else: 
        AUC  = metrics.roc_auc_score(target_list, prob_list)
        logger.info('===> Validation set: Average loss: {:.4f}\tAUC: {:.4f}\n'.format(
                    total_loss, AUC))

    conf_mat = metrics.confusion_matrix(target_list, pred_list)
    F1_score = metrics.f1_score(target_list, pred_list, average='weighted')

    # logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss, AUC, conf_mat, F1_score, target_list, pred_list 

