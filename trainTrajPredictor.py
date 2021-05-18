import sys

sys.path.append('../social_lstm')
from opTrajData import OpTrajData
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
# from social_lstm.model import SocialModel
import torch
from args import getArgs
from models import SocialTransformer, CoordLSTM, SocialLSTM, BGNLLLoss
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from utils import world2image, computeGlobalGroups, processGroups
import random


def train(epochs, device, loss, dloaders, checkPoint=False):
    if checkPoint:
        model = CoordLSTM(2, 32, device)  # SocialTransformer(2)#SocialModel(args)
        model.load_state_dict(torch.load('socLSTM.pt'))
        model = model.to(device)
        model = model.double()
    else:
        model = CoordLSTM(2,32,device)  # SocialTransformer(2)#SocialModel(args)
        model = model.to(device)
        model = model.double()
    opt = optim.RMSprop(model.parameters(), lr=5e-4)
    trainLoss = []
    validLoss = []
    random.shuffle(dloaders)
    print('Training on', [d.dataset.name for d in dloaders[:-1]])
    print('Validating on', dloaders[-1].dataset.name)
    for e in tqdm(range(epochs)):
        print("Epoch:", e)
        model.train()
        totalLoss = 0
        totLen = 0
        trackParams = []
        for dload in dloaders[:-1]:
            totLen += len(dload)
            model.h={}
            for peopleIDs, pos, target, ims in dload:
                # import pdb; pdb.set_trace()
                if len(pos[0][0])==len(pos[1][0]) and len(pos[1][0])==len(pos[2][0]) and len(pos[2][0])==len(pos[3][0]):#pos.size(1) > 0 and target.size(1) == pos.size(1):
                    # import pdb; pdb.set_trace()
                    # outputs=model(pos.double(),target.double())
                    gblGroups = []
                    for p in pos:
                        gblGroups.append(computeGlobalGroups(world2image(p[0], np.linalg.inv(data.H)), model.numGrids, model.gridSize))
                    groupedFeatures = processGroups(gblGroups, pos, model.h)
                    import pdb; pdb.set_trace()
                    # solution is to assume that its the same p people in al input frames so only using the people in the
                    # first peopleIDs list
                    # mux, muy, sx, sy, corr
                    coeffs = model(torch.tensor(peopleIDs[0]), torch.stack(pos).squeeze(1).double().to(device), torch.stack(groupedFeatures).to(device))
                    outputs = model.getCoords(coeffs) #these are the positions for when you want to plot those

                    l = loss(target,coeffs,peopleIDs)
                    # import pdb; pdb.set_trace()
                    opt.zero_grad()
                    if len(l)<=1:
                        l[0].backward()
                        totalLoss += l[0].item()
                    else:
                        for l_item in l:
                            l_item.backward(retain_graph=True)
                        totalLoss += torch.sum(torch.stack(l)).item()/len(l)
                    opt.step()
                    # trackParams.append([params[0].detach(), params[1].detach(), params[2].detach(), params[3].detach(), params[4].detach()])
                else:
                    totLen -= 1
        print('Train Loss:', totalLoss / totLen, 'totLen:', totLen)
        trainLoss.append(totalLoss / totLen)

        model.h={}
        l=test(device,loss,dloaders[-1],None,model)
        validLoss.append(l)
        print('Validation Loss:',l)
        print()
        torch.save(model.state_dict(), 'socLSTM.pt')
    # print(trainLoss)
    # print(validLoss)
    return trainLoss, validLoss



def test(device, loss, dloader, save_path, model=None):
    # test loop
    if model is None:
        model = CoordLSTM(2, 32, device)  # SocialTransformer(2)#SocialModel(args)
        model.load_state_dict(torch.load(save_path))
        model = model.to(device)
        model = model.double()
    model.eval()
    totalLoss = 0
    totLen = len(dloader)
    for peopleIDs, pos, target, ims in dloader:
        # import pdb; pdb.set_trace()
        if pos.size(1) > 0 and target.size(1) == pos.size(1):
            # outputs=model(pos.double(),target.double())
            gblGroups = []
            for p in pos:
                gblGroups.append(computeGlobalGroups(world2image(p, np.linalg.inv(data.H)), model.numGrids, model.gridSize))
            groupedFeatures = processGroups(gblGroups, pos, model.h)
            coeffs = model(torch.tensor(peopleIDs).to(device), pos.double().to(device),torch.stack(groupedFeatures).to(device))
            outputs = model.getCoords(coeffs)
            l = loss(target, coeffs, peopleIDs)
            if len(l) <= 1:
                totalLoss += l[0].item()
            else:
                totalLoss += torch.sum(torch.stack(l)).item() / len(l)
        else:
            totLen -= 1
    return totalLoss / totLen


dloaders=[]
for name in ['ETH','ETH_Hotel','UCY_Zara1','UCY_Zara2']:#['UCY_Zara2']:#
    data = OpTrajData(name, 'by_frame', 'mask')
    dloaders.append(DataLoader(data, batch_size=1, shuffle=False, drop_last=False))
loss = BGNLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 40

test_ind=random.choice(list(range(len(dloaders))))
test_set=dloaders.pop(test_ind)

trainLoss, validLoss = train(epochs, device, loss, dloaders, checkPoint=False)

print('Testing on',test_set.dataset.name)
testLoss = test(device, loss, test_set, 'socLSTM.pt')
print('Test Loss:', testLoss)

plt.plot(trainLoss)
plt.plot(validLoss)
plt.title('SLSTM Training Loss; Test on'+test_set.dataset.name+'='+str(testLoss)[:6])
plt.xlabel('Epoch')
plt.ylabel('Bivariate Gaussian NLL')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()