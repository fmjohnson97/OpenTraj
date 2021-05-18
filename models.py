import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv1d, Dropout, ReLU, LSTM, Transformer, TransformerEncoder, Linear
import numpy as np

class HumanTrajNet(nn.Module):
    def __init__(self):
        super(HumanTrajNet, self).__init__()

    def forward(self, x):
        pass


class SocialTransformer(nn.Module):
    def __init__(self, infeats):
        super(SocialTransformer, self).__init__()
        self.transf = Transformer(infeats, 1)

    def forward(self, x, y):
        return self.transf(x, y)


class SocialLSTM(nn.Module):
    def __init__(self, infeats):
        super(SocialLSTM, self).__init__()
        self.embed = Linear(infeats, 64)
        self.lstm = LSTM(64, 128)

    def forward(self, x):
        pass


class CoordLSTM(nn.Module):
    def __init__(self, infeats, socialFeats, device):
        super(CoordLSTM, self).__init__()
        self.coordEmbed = Linear(infeats, 32)
        self.socialEmbed = Linear(socialFeats, 32)
        self.outputEmbed = Linear(32, 5)
        self.lstm = LSTM(64, 32)
        self.relu = ReLU()
        self.h = {}
        self.device = device
        self.gridSize = (480, 640)
        self.numGrids = (3, 3)
        self.dropout = Dropout(0.1)

    def getHidden(self, personIDs):
        h = []
        c = []
        for p in personIDs:
            temp = self.h.get(p, (torch.rand(32), torch.rand(32)))
            h.append(temp[0])
            c.append(temp[1])
        return (torch.stack(h).unsqueeze(0).double().to(self.device), torch.stack(c).unsqueeze(0).double().to(self.device))

    def updateHidden(self, personIDs, h):
        for i, p in enumerate(personIDs):
            self.h[p.item()] = (h[0][0][i], h[1][0][i])

    def forward(self, peopleIDs, x, gblTensor):
        x = self.dropout(self.relu(self.coordEmbed(x)))
        gblTensor = self.dropout(self.relu(self.socialEmbed(gblTensor.double())))
        h = self.getHidden(peopleIDs)
        try:
            x, h = self.lstm(torch.cat((x, gblTensor), -1), h)
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()

        self.updateHidden(peopleIDs, (h[0].detach(), h[1].detach()))
        x = self.outputEmbed(x)
        return x

    def getCoords(self, output):
        # modified from https://github.com/quancore/social-lstm
        # import pdb;pdb.set_trace()
        mux, muy, sx, sy, corr = output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3], output[:, :, 4]
        sx = torch.exp(sx)
        sy = torch.exp(sy)
        corr = torch.tanh(corr)

        coords = []
        for batch in range(output.shape[0]):
            o_mux, o_muy, o_sx, o_sy, o_corr = mux[batch, :], muy[batch, :], sx[batch, :], sy[batch, :], corr[batch, :]
            numNodes = o_mux.shape[0]
            next_x = torch.zeros(numNodes)
            next_y = torch.zeros(numNodes)

            for node in range(numNodes):
                mean = [o_mux[node], o_muy[node]]
                cov = [[o_sx[node] * o_sx[node], o_corr[node] * o_sx[node] * o_sy[node]],
                       [o_corr[node] * o_sx[node] * o_sy[node], o_sy[node] * o_sy[node]]]
                mean = np.array(mean, dtype='float')
                cov = np.array(cov, dtype='float')
                next_values = np.random.multivariate_normal(mean, cov, 1)
                next_x[node] = next_values[0][0]
                next_y[node] = next_values[0][1]
            coords.append(np.concatenate((next_x.reshape(-1, 1), next_y.reshape(-1, 1)),1))
        # import pdb; pdb.set_trace()
        # return torch.cat((next_x.reshape(-1, 1), next_y.reshape(-1, 1)), -1), [mux.squeeze(0), muy.squeeze(0), sx.squeeze(0), sy.squeeze(0), corr.squeeze(0)]
        return coords


class BGNLLLoss(nn.Module):
    def __init__(self):
        super(BGNLLLoss, self).__init__()

    def forward(self, targets, params, peopleIDs):
        # modified from https://github.com/quancore/social-lstm
        import pdb; pdb.set_trace()
        mux, muy, sx, sy, corr = params[:, :, 0], params[:, :, 1], params[:, :, 2], params[:, :, 3], params[:, :, 4]
        sx = torch.exp(sx)
        sy = torch.exp(sy)
        corr = torch.tanh(corr)

        #TODO: How to fix the size mis match?? have to compute loss per person then...
        batchLoss=[]
        for batch in range(len(targets)):
            normx = targets[batch][:, :, 0].squeeze(0) - mux[batch]
            normy = targets[batch][:, :, 1].squeeze(0) - muy[batch]
            sxsy = sx[batch] * sy[batch]

            z = (normx / sx[batch]) ** 2 + (normy / sy[batch]) ** 2 - 2 * (corr[batch] * normx * normy / sxsy)
            negRho = 1 - corr[batch] ** 2
            result = torch.exp(-z / (2 * negRho)) / (2 * np.pi * (sxsy * torch.sqrt(negRho)))
            epsilon = 1e-20
            result = -torch.log(torch.clamp(result, min=epsilon))

        loss = [0] * len(peopleIDs)
        counter = 0
        for frame in range(targets.shape[0]):
            for person in range(targets.shape[1]):
                loss[person] += result[(frame, person)]
                counter = counter + 1

        # TODO: FIX THIS WHEN CHANGE TO LARGER WINDOW SIZE
        # if counter != 0:
        #     return loss / counter
        # else:
        #     return loss
        return loss
