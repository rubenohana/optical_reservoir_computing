import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import lightonml
from lightonml import OPU
from lightonml.encoding import base


class ESN(torch.nn.Module):
    """
    Implements an Echo State Network.
    Parameters:
      - input_size: dimension of the input
      - res_size: number of units in the reservoir
      - random_projection: "gpu" for GPU, "opu" for OPU
      - input_scale: scale of the input-to-reservoir matrix
      - res_scale: scale of the reservoir weight matrix
      - bias_scale: scale of the random additive bias
      - leak_rate: leak rate to control the dynamics (1 = conventional RC)
      - f: activation function of the reservoir
      - redraw: if True, redraw matrices at each iteration
      - seed: random seed for reproducibility of the results
      - device: GPU or CPU
      - precision: number of bit plans sent to the OPU (because its input needs to be binary)
    """

    def __init__(self, input_size, res_size, device ,
                 random_projection = 'gpu', 
                 input_scale = 1.0, res_scale = 1.0, 
                 bias_scale = 0.4, leak_rate = 1, f = 'erf', 
                 redraw = False, 
                 seed = 1, precision = 8):
        super(ESN, self).__init__()

        self.input_size = input_size
        self.res_size = res_size
        self.input_scale = input_scale
        self.res_scale = res_scale
        self.leak_rate = leak_rate
        self.redraw = redraw
        self.random_projection = random_projection
        self.seed = seed
        self.device = device
        self.precision = precision
        self.bias_scale = bias_scale
        torch.manual_seed(self.seed)
        self.bias = self.bias_scale * torch.randn(self.res_size).to(self.device)
        # Some preprocessing
        if f == 'erf':
            self.f = torch.erf
        if f == 'cos_rbf':
            torch.manual_seed(1)
            self.bias = 2 * np.pi * torch.rand(self.res_size).to(self.device)
            self.f = lambda x: np.sqrt(2)*torch.cos(x + self.bias)
        if f == 'heaviside':
            self.f = lambda x: 1 * (x > 0)
        if f == 'sign':
            self.f = torch.sign
        if f == 'linear':
            self.f = lambda x: x
        if f == 'relu':
            self.f = torch.relu
        if f == 'intensity':
            self.f = lambda x : torch.abs(x)
                
        # Generation of the weights of the reservoir
        torch.manual_seed(self.seed)
        if self.random_projection == 'gpu':
            self.W_in = torch.randn(res_size, input_size).to(self.device)
            self.W_res = torch.randn(res_size, res_size).to(self.device)
            
    def OPU_step(self, input_data, opu, encoder = None, decoder = None):        
        output_OPU = opu.linear_transform(input_data.repeat(2,1), encoder, decoder)
        #return output_OPU[0,:]
        return torch.mean(output_OPU, 0)
    
    def calibrate_std_OPU(self, opu, encoder, decoder):
        np.random.seed(0)
        M = np.random.randn(300, self.res_size + self.input_size)
        M = M / np.linalg.norm(M, axis = 1).reshape(-1,1) #sending vectors of norm 1
        opu.fit1d(M)
        std = opu.linear_transform(M, encoder, decoder).std()
        del M
        return std
    
        
    def forward(self, input_data, initial_state=None):
        """
        Compute the reservoir states for the given sequence.
        Parameters:
          - input: Input sequence of shape (seq_len, input_size), i.e. (t,d)
        
        Returns: a tensor of shape (seq_len, res_size)
        """
        seq_len = input_data.shape[0]
        d = input_data.shape[1]
        
        x = torch.zeros((seq_len, self.res_size)).to(self.device)  # will contain the reservoir states
        #x = torch.zeros(( self.res_size)).to(self.device)
        if initial_state is not None:
            x[-1, :] = initial_state
            #x = initial_state
        #print(x.shape)
        
        if self.random_projection == 'opu':
            # Calibrating the OPU Gaussian distribution to obtain a Standard distribution
            encoder = base.SeparatedBitPlanEncoder(precision = self.precision)
            decoder = base.SeparatedBitPlanDecoder
            opu_forward = OPU(n_components = self.res_size)
            std = self.calibrate_std_OPU(opu_forward, encoder,decoder) # obtaining the standard deviation of the OPU projection to standardize its distribution
            
        for i in tqdm(range(seq_len)):
            if not self.redraw:
                if self.random_projection == 'gpu':
                    s = torch.clone(x[i-1, :]).to(self.device)
                    proj_res = self.res_scale * self.W_res @ s
                    #proj_res = self.res_scale * self.W_res @ x
                    proj_input =self.input_scale * self.W_in @ input_data[i, :]
                    
                    x[i,:] = (1 - self.leak_rate) * s + self.leak_rate * self.f(proj_res + proj_input + self.bias) / np.sqrt(self.res_size)
                    #x = (1 - self.leak_rate) * x + self.leak_rate * self.f(proj_res + proj_input + self.bias) / np.sqrt(self.res_size)

                elif self.random_projection == 'opu':
                    if self.device ==torch.device("cuda"):
                        to_proj = torch.cat((self.res_scale/std * x[i-1, :], self.input_scale/std * input_data[i, :])).cpu()
                        opu_proj = self.OPU_step(to_proj, opu_forward, encoder, decoder).to(self.device) + self.bias
                        x[i,:] = (1 - self.leak_rate) * x[i-1, :] + self.leak_rate * self.f(opu_proj) / np.sqrt(self.res_size)
                    else:
                        to_proj = torch.cat((self.res_scale/std * x[i-1, :], self.input_scale/std * input_data[i, :])).cpu()
                        opu_proj = self.OPU_step(to_proj, opu_forward, encoder, decoder) + self.bias
                        x[i,:] = (1 - self.leak_rate) * x[i-1, :] + self.leak_rate * self.f(opu_proj) / np.sqrt(self.res_size)
        
        if self.random_projection == 'opu':
            opu_forward.close()
            
        return x

    def train(self, X, y, method='cholesky', alpha=1e-2, lr = 1e-2, epochs = 1000):
        """
        Compute the output weights with a linear regression.
        Parameters:
          - X: input sequence of shape (seq_len, res_size)
          - y: target output (seq_len, out_dim)
          - method: "cholesky" or "sklearn ridge"
          - alpha: L2-regularization parameter
        
        Returns: a tensor of shape (res_size, out_dim)
        """
        if method == 'cholesky':
            # This technique uses the Cholesky decomposition
            # It should be fast when res_size < seq_len
            Xt_y = X.T @ y  # size (res_size, out_dim)
            K = X.T @ X  # size (res_size, res_size)
            K.view(-1)[::len(K)+1] += alpha  # add elements on the diagonal inplace
            L = torch.cholesky(K, upper=False)
            return torch.cholesky_solve(Xt_y, L, upper=False)
        elif method == 'sklearn ridge':
            from sklearn.linear_model import Ridge
            clf = Ridge(fit_intercept=False, alpha=alpha)
            clf.fit(X.cpu().numpy(), y.cpu().numpy())
            return torch.from_numpy(clf.coef_.T).to(self.device)
        
        elif method == 'sgd':
            model = nn.Linear(X.shape[1],y.shape[1], bias = True).to(self.device)
            criterion = nn.MSELoss()
            opt = optim.Adam(model.parameters(),lr=lr, weight_decay = alpha)#, lambd = alpha/100)#,weight_decay = alpha)#,lambd = 0.01)
            epoch = int(epochs)
            for i in tqdm(range(epoch)):
                opt.zero_grad()
                output = model(X)
                loss = criterion(output,y)#(model(torch.randn(100,1)), torch.randn(100,1))
                #if i % (epoch/10) == 0:
                #    print(loss)
                loss.backward()
                opt.step()
            loss.detach()
            return model

    def rec_pred(self, X, output_w, n_rec, input_dim, concat=None, renorm_factor=None):
        """
        Performs a recursive prediction.
        Parameters:
        - X: reservoir states of shape (seq_len, res_size)
        - output_w: output weights of shape (res_size, out_dim)
        - n_rec: number of recursive predictions (0 = standard non-rec prediction)
        - input_dim: dimension of the input (= self.input_dim, could be removed)
        - concat: if not None, concatenate these input states with the current reservoir state
        - renorm_factor: gives the renorm factor for the concat
        
        Returns: a tensor of shape (input_len, out_dim*(n_rec+1))
        """
        #Xd = X.to(self.device)
        if self.random_projection == 'gpu':
            Xd = X[-1,:].type(torch.FloatTensor).reshape(-1,1).to(self.device)
            concat = concat[-1,:].type(torch.FloatTensor).reshape(-1,1).to(self.device)
            #output_w = output_w.type(torch.FloatTensor).to(self.device)
        elif self.random_projection == 'opu':
            Xd = X[-1,:].type(torch.DoubleTensor).reshape(-1,1).to(self.device)
            concat = concat[-1,:].type(torch.DoubleTensor).reshape(-1,1).to(self.device)# input_len, n_res
            output_w = output_w.type(torch.DoubleTensor).to(self.device)
        input_len = X.shape[0]
        out_len = self.input_size #output_w.shape[1]
        d = input_dim
        
        total_pred = torch.zeros(1, out_len*(n_rec+1)).to(self.device) # 1000 x (1000*200)
        try:
            output_w = output_w.weight.data.type(torch.FloatTensor).T.to(self.device)
        except:
            pass
        
        # Prediction step
        if concat is None:
            pred_data = Xd @ output_w
        else:
            pred_data = torch.cat((Xd, concat), dim=0).T @ output_w #1 x 8192 @ 8192 x 1000 #dim was 1 before X[1,:]
        
        total_pred[0, :out_len] = pred_data # 1x1000
        single_pred_length = out_len // input_dim # = 1 

        if self.random_projection == 'opu':
        # Calibrating the OPU Gaussian distribution to obtain a Standard distribution
            encoder = base.SeparatedBitPlanEncoder(precision = self.precision)
            decoder = base.SeparatedBitPlanDecoder
            opu_forward = OPU(n_components = self.res_size)
            std = self.calibrate_std_OPU(opu_forward, encoder,decoder) # obtaining the standard deviation of the OPU projection to standardize its distribution
            
        for i_rec in tqdm(range(n_rec)):
            for t in range(single_pred_length):
                # The next input corresponds to the previous prediction
                pred_input = pred_data[:, t*input_dim:(t+1)*input_dim].to(self.device) #1x1000 # input_len, input_dim
                # Forward step
                if self.random_projection == 'gpu':
                    proj_res = self.res_scale * Xd.to(self.device).T @ self.W_res.to(self.device).T
                    #print(proj_res.shape)
                    proj_input = self.input_scale * pred_input.to(self.device) @ self.W_in.to(self.device).T
                    #print(proj_input.shape)
                    bias = self.bias.reshape(-1,1).to(self.device)
                    #print("bias",bias.shape)
                    #print("xd",Xd.shape)
                    Xd = (1 - self.leak_rate) * Xd + self.leak_rate * self.f( proj_res.T + proj_input.T + bias) / np.sqrt(self.res_size)
                    #print(Xd.shape)
                        
                elif self.random_projection == 'opu':
                    if self.device ==torch.device("cuda"):
                        to_proj = torch.cat((self.res_scale/std * Xd, self.input_scale/std * pred_input.T)).cpu().ravel().T
                        opu_proj = self.OPU_step(to_proj, opu_forward, encoder, decoder).to(self.device)
                        Xd = (1 - self.leak_rate) * Xd + self.leak_rate * self.f(opu_proj + self.bias).reshape(-1,1)/ np.sqrt(self.res_size)

                    else:
                        to_proj = torch.cat((self.res_scale/std * Xd, self.input_scale/std * pred_input.T)).ravel().T
                        opu_proj = self.OPU_step(to_proj,opu_forward, encoder, decoder)
                        Xd = (1 - self.leak_rate) * Xd + self.leak_rate * self.f(opu_proj + self.bias).reshape(-1,1)/ np.sqrt(self.res_size)

            # Prediction step
            if concat is None:
                pred_data = Xd @ output_w
            else:
                #print(pred_data.shape)
                
                concat = pred_data[:, -input_dim:] * renorm_factor
                if self.random_projection == 'gpu':
                    pred_data = torch.cat((Xd, concat.T), dim=0).T @ output_w
                if self.random_projection == 'opu':
                    pred_data = torch.cat((Xd, concat.T), dim=0).T @ output_w
                #pred_data = torch.cat((Xd, concat.T), dim=0).T @ output_w
            total_pred[:, (i_rec+1)*out_len:(i_rec+2)*out_len] = pred_data
        
        if self.random_projection == 'opu':
            opu_forward.close()
        
        return total_pred