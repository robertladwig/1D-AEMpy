import numpy as np
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt

class Utils:
    
    def __init__(self, num_features, inp_cols, target_cols, date_col, input_window, output_window, num_out_features, stride=1):
        self.num_features = num_features
        self.inp_cols = inp_cols
        self.target_cols = target_cols
        self.date_col = date_col
        self.input_window = input_window
        self.output_window = output_window
        self.num_out_features = num_out_features
        self.stride = stride
        self.y_mean = None
        self.y_std = None
    
    def train_test_split(self, df, split_type='ratio', split_date=None, split_ratio=0.0):
        '''

        split time series into train/test sets

        : param df:                     time array
        : para y:                       feature array
        : para split:                   percent of data to include in training set
        : return t_train, y_train:      time/feature training and test sets;
        :        t_test, y_test:        (shape: [# samples, 1])

        '''
        if split_type == 'time':
            df_train = df[df[self.date_col] <= split_date]
            df_test = df[df[self.date_col] > split_date]
            return df_train, df_test
        else:
            indx_split = int(split_ratio * df.shape[0])
            indx_train = np.arange(0, indx_split)
            indx_test = np.arange(indx_split, df.shape[0])
    
            df_train = df.iloc[indx_train]
            df_test = df.iloc[indx_test]

        return df_train.reset_index(drop='true'), df_test.reset_index(drop='true')

#     def normalize(self, df):
#         '''
#         Normalize data
#         '''
        
#         # compute mean and std of target variable - to be used for unnormalizing
#         self.y_std = df[self.target_cols].std()[0]
#         self.y_mean = df[self.target_cols].mean()[0]
        
#         if len(set(self.inp_cols).intersection(self.target_cols))==0:
#             df[self.inp_cols] = (df[self.inp_cols]-df[self.inp_cols].mean())/df[self.inp_cols].std()
#             df[self.target_cols] = (df[self.target_cols]-self.y_mean)/self.y_std
#         else:
#             df[self.inp_cols] = (df[self.inp_cols]-df[self.inp_cols].mean())/df[self.inp_cols].std()
            
#         return df
    
    def normalize(self, df, use_stat=False):
        '''
        Normalize data
        '''
        if use_stat:
            if len(set(self.inp_cols).intersection(self.target_cols))==0:
                df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
                df[self.target_cols] = (df[self.target_cols]-self.y_mean)/self.y_std
            else:
                df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            return df
                
            # compute mean and std of target variable - to be used for unnormalizing
        self.y_std = df[self.target_cols].std()[0]
        self.y_mean = df[self.target_cols].mean()[0]
         
        if len(set(self.inp_cols).intersection(self.target_cols))==0:
            
            self.feat_mean = df[self.inp_cols].mean()
            self.feat_std = df[self.inp_cols].std()
            
            df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            df[self.target_cols] = (df[self.target_cols]-self.y_mean)/self.y_std
        else:
            self.feat_mean = df[self.inp_cols].mean()
            self.feat_std = df[self.inp_cols].std()
            df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            
        return df
        
    def windowed_dataset(self, df):
        '''
        create a windowed dataset
    
        : param y:                time series feature (array)
        : param input_window:     number of y samples to give model
        : param output_window:    number of future y samples to predict
        : param stide:            spacing between windows
        : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
        : return X, Y:            arrays with correct dimensions for LSTM
        :                         (i.e., [input/output window size # examples, # features])
        '''

        L = df.shape[0]
        num_samples = (L - self.input_window - self.output_window) // self.stride + 1
    
        dfX = df[self.inp_cols]
        dfY = df[self.target_cols]
        
        X = np.zeros([num_samples, self.input_window, self.num_features])
        Y = np.zeros([num_samples, self.output_window, self.num_out_features])
        # target_X = np.zeros([self.input_window, num_samples, self.num_out_features])
        
        # shuffled_inds = random.sample(range(num_samples),num_samples)
        
        for ii in np.arange(num_samples):
            start_x = self.stride * ii
            end_x = start_x + self.input_window
            X[ii, :, :] = dfX.iloc[start_x:end_x, :]
            
            start_y = self.stride * ii + self.input_window
            end_y = start_y + self.output_window
            Y[ii, :, :] = dfY.iloc[start_y:end_y, :]

        return X, Y
   

    def numpy_to_torch(self, Xtrain, Ytrain, Xtest, Ytest):
        '''
        convert numpy array to PyTorch tensor
    
        : param Xtrain:               windowed training input data (# examples, input window size, # features); np.array
        : param Ytrain:               windowed training target data (# examples, output window size, # features); np.array
        : param Xtest:                windowed test input data (# examples, input window size, # features); np.array
        : param Ytest:                windowed test target data (# examples, output window size, # features); np.array
        : return X_train_torch, Y_train_torch,
        :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

        '''

        X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
        Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

        X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
        Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

        return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
    
    
    def plot(self, df, pred_y, true_y, idx):
        '''
        param:
        
        Return:
        '''
        input_window = self.input_window
        output_window = self.output_window
        df_un = df.copy(deep=True)
        
        df_un[self.target_cols] = df_un[self.target_cols].apply(lambda l: l*self.y_std+self.y_mean)
        df_un = df_un.reset_index(drop=True)
        
        x_axis_1 = df_un.loc[idx:idx+input_window-1,self.date_col].to_numpy().reshape(-1)
        
        x_axis_2 = df_un.loc[idx+input_window:idx+input_window+output_window-1,self.date_col].to_numpy().reshape(-1)
        
        x_plot = df_un[self.target_cols].to_numpy()[idx:idx+input_window].reshape(-1)
        
        pred_plot = pred_y[idx].cpu().numpy().reshape(-1)
        
        true_plot = true_y[idx].cpu().numpy().reshape(-1)
    
        plt.figure(figsize=(20,4), dpi=150)
        plt.grid("on", alpha=0.4)
        plt.plot(x_axis_1, x_plot, marker='o', linestyle='--', label='input_window')
        plt.plot(x_axis_2, pred_plot, marker='o', linestyle='--', label='predictions')
        plt.plot(x_axis_2, true_plot, marker='o', linestyle='--', label='true_values')
        plt.xlabel('Timeline')
        plt.ylabel('Chlorophyll')
        plt.xticks(rotation=90)
        plt.legend()
    
    
    def plot_samples(self, df, pred_y, true_y, idx, n_samples, xticks_spacing=False):
        '''
        params:
        
        return:
        '''
    
        input_window = self.input_window
        output_window = self.output_window
        df_un = df.copy(deep=True)
        
        df_un[self.target_cols] = df_un[self.target_cols].apply(lambda l: l*self.y_std+self.y_mean)
        df_un = df_un.reset_index(drop=True)
        
        x_axis_1 = df_un.loc[idx:idx+input_window-1,self.date_col].to_numpy().reshape(-1)
        
        start_x_axis = idx + input_window
        delta = n_samples*output_window - 1 if n_samples*output_window - 1 < (df_un.shape[0]-start_x_axis) else df_un.shape[0]-start_x_axis
        
        end_x_axis = start_x_axis + delta #n_samples*output_window - 1
        
        x_axis_2 = df_un.loc[start_x_axis:end_x_axis,self.date_col].to_numpy().reshape(-1)
            
        x_plot = df_un[self.target_cols].to_numpy()[idx:idx+input_window].reshape(-1)
        
        pred_plot = pred_y[idx:idx+n_samples*output_window:output_window].cpu().numpy().reshape(-1)
        true_plot = true_y[idx:idx+n_samples*output_window:output_window].cpu().numpy().reshape(-1)
    
        fig,ax = plt.subplots()
        
        fig.set_figheight(5)
        fig.set_figwidth(20)

        ax.grid(visible=True, alpha=0.2)
        ax.plot(x_axis_1, x_plot, linestyle='--', label='input_window')
        ax.plot(x_axis_2, pred_plot, linestyle='--', label='predictions')
        ax.plot(x_axis_2, true_plot, linestyle='--', label='true_values')
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Chlorophyll')
        
        if xticks_spacing:
            every_nth = 20
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)

        ax.set_xticklabels(x_axis_1, rotation=90)
        ax.set_xticklabels(x_axis_2, rotation=90)
        plt.legend()
        
    
    def pred_per_step_helper(self, predictions, idx, pred_values, date):
        '''
        Compute all the predictions for a single date. i.e. as T+1, T+2, T+3, ... T+horizon timestep prediction
        '''
        c = 0
        rind = idx
        while c<self.output_window:
            rind = rind + c
            pred_values[date].append(predictions[rind].reshape(-1)[-(c+1)])
            c+=1
    
        return pred_values

    def prediction_per_step(self, df, predictions, gts, ids):
        pred_values = {}
        gts_ls = {}
        for idx in ids:
            date = df.loc[idx+self.input_window+self.output_window-1,self.date_col]
            pred_values[date] = []
            gts_ls['GT_'+date] = gts[idx].reshape(-1)[-1]
            pred_values = self.pred_per_step_helper(predictions, idx, pred_values, date)
    
        pred_values = {k:list(reversed(v)) for k,v in pred_values.items()}
        for k,v in pred_values.items():
            pred_values[k] = [i.cpu().numpy() for i in v]
            
        return pred_values, gts_ls
    
    
    def plot_time_step_predictions(self, pred_values, gt):
        
        """
        Currently the number of plots is hard-configured to 6
        """
        x_axis_label = 'T+'
        x_axis = [x_axis_label+str(i) for i in range(1,self.output_window+1)]
        
        fig, axes = plt.subplots(2, 3, figsize=(25, 15), dpi=150)
        for i,v in enumerate(pred_values.items()):
            ax = axes[i//3, i%3]
            ax.grid(alpha=0.7)
            ax.plot(x_axis, v[1], marker='o', linestyle='dashed', linewidth=2, markersize=12)
            gt_ind_label = 'GT_'+v[0]
            ax.axhline(gt[gt_ind_label].cpu().numpy())
            ax.set_xlabel(f"For date = {v[0]}, ground-truth was = {gt[gt_ind_label]}")
            ax.set_ylabel('Predicted values')
    
    def fillpredtable(self, r, table, pred):
        for i,k in enumerate(table.columns):
            if i-(r-1)>=0 and i-(r-1) < pred.shape[0]:
                table.loc[r, k] = pred[i-(r-1)][r-1].cpu().numpy()
        return table
            
    def predictionTable(self, df, pred_df, gt_values=None, plot=True):
        '''
        Create the prediction table
        '''
        pred_table = np.zeros((self.output_window, df.shape[0] - self.input_window))
        pred_table = pd.DataFrame(pred_table)
        pred_table.columns = df[self.input_window:].Date.values
        pred_table.index = range(1,self.output_window+1)
        pred_table.loc[:] = np.nan
        
        for r in range(1, self.output_window+1):
            pred_table = self.fillpredtable(r, pred_table, pred_df)

        if plot:    
            plot_df = pred_table.iloc[:, self.output_window-1:-self.output_window+1]
            plot_gt_values = gt_values[self.output_window-1:-self.output_window+1]
            return pred_table, plot_df, plot_gt_values

        return pred_table

    def plotTable(self, plot_df, plot_gt, T):
        '''
        Plot the prediction table
        '''
        x_plot = plot_df.columns
        print(x_plot)
        fig,ax = plt.subplots()
        
        fig.set_figheight(5)
        fig.set_figwidth(20)
        
        ax.grid(visible=True, alpha=0.2)
        
        for t in T:
            y_axis = plot_df.loc[t,:].values
            ax.plot(x_plot, y_axis, linestyle='--', label='T+'+str(t))
        
        ax.plot(x_plot, plot_gt, linestyle='--', label='Ground-truth')
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Chlorophyll T+n predictions')
            
        every_nth = 20
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        ax.set_xticklabels(x_plot, rotation=90)
        plt.legend()
    
    def compute_rmse(self, i, ptable, gt_values):
    
        tk = ptable.iloc[i,:].values
        null_inds = np.where(np.isnan(tk))[0]
        mask = np.ones(gt_values.shape)
        mask[null_inds]= 0
        tk = np.nan_to_num(tk)
        
        unreduced_loss = (tk-gt_values)**2
        unreduced_loss = (unreduced_loss * mask).sum()
        
        non_zero_elements = mask.sum()
        loss = unreduced_loss / non_zero_elements
        
        rmse = loss**0.5
        return rmse