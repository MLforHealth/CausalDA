import torch
import logging
import os
import torch.nn.functional as F

logger = logging.getLogger("causal_bootstrap")

def cost_fn(out, tar, weighting):
    return F.cross_entropy(out, tar, weight=weighting)

def train_step(data, target, extras, args, model, writer, device, optimizer, batch_size):
    model.train() 
    
    data = data.float().to(device) # add channel dimension
    target = target.long().to(device)
    target = target.view((-1))
    
    if torch.is_tensor(extras):
        extras = extras.float().to(device)

    output = model(data, extras) ## no softmax applied 
    weights = None

    loss = cost_fn(output, target, weighting=weights)

    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

    pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability

    acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)
        
    return loss.item(), acc


def train(args, model, writer, device, train_loader, optimizer, epoch, batch_size):
    model.train() # training model 

    epoch_loss = 0
    epoch_acc = 0

    offset = len(train_loader) * (epoch-1)
    
    for batch_idx, [data, target, extras ] in enumerate(train_loader):        
        step_loss, step_acc = train_step(data, target, extras, args, model, writer, device, optimizer, batch_size)

        epoch_loss += step_loss
        epoch_acc += step_acc
        
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), step_acc, step_loss))
        
        # index.append((batch_idx+offset))
        writer.add_scalar('Loss/train/batch', step_loss, (batch_idx+offset))
        writer.add_scalar('Accuracy/train/batch', step_acc, (batch_idx+offset))

    return (epoch_loss/len(train_loader)), (epoch_acc/len(train_loader))

def snapshot(dir_path, res_pth, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

    with open(os.path.join(res_pth, "best_model_pth.txt"), 'w') as res:
        res.write(f"Best epoch: {state['epoch']}, Best val acc: {state['validation_acc']}, Best val loss: {state['validation_loss']}, Model path: {snapshot_file}")
