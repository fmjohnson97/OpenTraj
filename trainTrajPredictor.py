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


def train(epochs, device, loss, dloader):
    model = CoordLSTM(2,device)  # SocialTransformer(2)#SocialModel(args)
    model = model.to(device)
    model = model.double()
    opt = optim.RMSprop(model.parameters(), lr=5e-4)
    trackLoss = []
    for e in range(epochs):
        print("Epoch:", e)
        totalLoss = 0
<<<<<<< Updated upstream
        totLen = len(dloader)
        for peopleIDs, pos, target, ims in tqdm(dloader):
            pos=torch.stack(pos)
            import pdb; pdb.set_trace()
            # if pos.size(1) > 0 and target.size(1) == pos.size(1):
                # outputs=model(pos.double(),target.double())
            gblGroups = []
            for p in pos:
                gblGroups.append(computeGlobalGroups(world2image(p[0], data.H), model.numGrids, model.gridSize))
            groupedFeatures = processGroups(gblGroups, pos, 'coords')
            coeffs = model(torch.tensor(peopleIDs[:len(pos)]).to(device), pos.squeeze(1).double().to(device), torch.stack(groupedFeatures).squeeze(1).to(device))
            outputs, params = model.getCoords(coeffs)
            l = loss(target,params)
            # import pdb; pdb.set_trace()
            opt.zero_grad()
            l.backward()
            opt.step()
            totalLoss += l.item()
            # else:
            #     totLen -= 1
        print('Loss:', totalLoss / totLen, 'totLen:', totLen)
        trackLoss.append(totalLoss / totLen)
    torch.save(model.state_dict(), 'coordLSTMweights.pt')
    print(trackLoss)
    try:
        plt.plot(trackLoss)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Bivariate Gaussian NLL')
        plt.show()
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
=======
        totLen=0
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
                    # import pdb; pdb.set_trace()
                    # solution is to assume that its the same p people in al input frames so only using the people in the
                    # first peopleIDs list
                    coeffs = model(torch.tensor(peopleIDs[0]), torch.stack(pos).squeeze(1).double().to(device), torch.stack(groupedFeatures).to(device))
                    outputs, params = model.getCoords(coeffs)
                    # mux, muy, sx, sy, corr
                    l = loss(target,params,peopleIDs)
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

>>>>>>> Stashed changes


def test(device, loss, dloader, save_path):
    # test loop
    model = CoordLSTM(2,device)  # SocialTransformer(2)#SocialModel(args)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    model = model.to(device)
    model = model.double()
    totalLoss = 0
    totLen = len(dloader)
    for peopleIDs, pos, target, ims in tqdm(dloader):
        # import pdb; pdb.set_trace()
        # if pos.size(1) > 0 and target.size(1) == pos.size(1):
            # outputs=model(pos.double(),target.double())
        gblGroups = []
        for p in pos:
            gblGroups.append(computeGlobalGroups(world2image(p[0], data.H), model.numGrids, model.gridSize))
        groupedFeatures = processGroups(gblGroups, pos, 'coords')
        coeffs = model(peopleIDs, pos.double(), torch.stack(groupedFeatures))
        outputs, params = model.getCoords(coeffs)
        l = loss(target, params)
        totalLoss += l.item()
        # else:
        #     totLen -= 1
    print('Loss:', totalLoss / totLen, 'totLen:', totLen)


# args=getArgs()
data = OpTrajData('ETH', 'by_frame', 'mask')
dloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=False)
loss = BGNLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 40

<<<<<<< Updated upstream
train(epochs, device, loss, dloader)
=======
test_ind=random.choice(list(range(len(dloaders))))
test_set=dloaders.pop(test_ind)

trainLoss, validLoss = train(epochs, device, loss, dloaders)#, checkPoint=True)

print('Testing on',test_set.dataset.name)
testLoss = test(device, loss, test_set, 'socLSTM.pt')
print('Test Loss:', testLoss)
>>>>>>> Stashed changes

data = OpTrajData('UCY', 'by_frame', None)
dloader = DataLoader(data, batch_size=1, shuffle=False, drop_last=False)
test(device, loss, dloader, 'coordLSTMweights.pt')
