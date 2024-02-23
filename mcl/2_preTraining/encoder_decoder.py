import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim
from datetime import timedelta, datetime
import copy
import wandb
import math
import matplotlib.pyplot as plt

wandb.login()

from utils import Utils


class encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1, model_type='LSTM', dropout=0.0):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.model_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x_input):

        '''
        : param x_input:               input of shape (# in batch, seq_len, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        lstm_out, self.hidden = self.model(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[0]
        : return:              zeroed hidden state and cell state
        '''
        if self.model_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1, model_type='LSTM', dropout=0.0):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.model_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

        # TODO: predict mean and max
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, self.hidden = self.model(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))

        return output, self.hidden

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, features, target):
        'Initialization'
        self.features = features
        self.target = target

    def __len__(self):
        'Denotes the total number of samples'
        return self.features.__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.features[index]
        y = self.target[index]

        return X, y

class EarlyStopping:
    
    def __init__(self, thres=2, min_delta=0):
        
        self.thres = thres
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.thres:
                return True
        else:
            self.counter -= 1
            if self.counter < 0:
                self.counter = 0

        return False


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, out_path, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.out_path = out_path
        
    def __call__(
        self, current_valid_loss, model, epoch, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), self.out_path)
            
class seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, 
                 hidden_size, 
                 output_size=1, 
                 model_type='LSTM', 
                 num_layers=2, 
                 utils=None, 
                 dropout=0.0,
                 device=torch.device("cpu")):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of lstm in both encoder and decoder
        '''

        super(seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.output_size = output_size
        self.device = device

        self.encoder = encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, model_type=model_type, dropout=dropout).to(device)
        self.decoder = decoder(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, model_type=model_type, dropout=dropout).to(device)
        
        self.encoder_init = copy.deepcopy(self.encoder)
        self.decoder_init = copy.deepcopy(self.decoder)
        
        self.utils = utils
        
        
    def train_model(self, 
                    X_train, 
                    Y_train, 
                    X_test,
                    Y_test,
                    target_len,
                    config,
                    project_name,
                    run_name,
                    save_code,
                    training_prediction='recursive', 
                    dynamic_tf=False):

        '''
        train lstm encoder-decoder

        : param X_train:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param Y_train:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict. Time horizon
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        wandb.init(project=project_name, name=run_name, config=config, save_code=save_code)
        config= wandb.config
        
        n_epochs = config.epochs
        
        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        test_rmse = []
        train_rmse = []
        
        n_batches = int(math.ceil(X_train.shape[0] / config.batch_size))
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.max_lr, epochs=n_epochs, div_factor=config.div_factor, 
                                                        pct_start=config.pct_start, anneal_strategy=config.anneal_strategy, final_div_factor=config.final_div_factor,
                                                        steps_per_epoch=n_batches, verbose=False)
        early_stop = config.early_stop
        early_stopper = EarlyStopping(thres=config.early_stop_thres, min_delta=config.early_stop_delta)
        
        params = {
                  'batch_size': config.batch_size,
                  'shuffle': config.batch_shuffle
                }
        
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        # Generators
        training_set = Dataset(X_train, Y_train)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        
        validation_set = Dataset(X_test, Y_test)
        validation_generator = torch.utils.data.DataLoader(validation_set, **params)
        
        wandb.watch(self.encoder)
        
        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                # batch_test_loss = np.nan
                
                encoder_hidden = self.encoder.init_hidden(config.batch_size)

                for input_batch, target_batch in training_generator:
                   
                    # outputs tensor
                    outputs = torch.zeros(target_batch.shape[0], target_batch.shape[1], target_batch.shape[2], device=self.device)
                    # initialize hidden state
                    #encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    
                    # decoder with teacher forcing
                    # TODO: first input to decoder - shape: (batch_size, input_size)
                    decoder_input = torch.zeros([target_batch.shape[0], target_batch.shape[2]], device=self.device)  # shape: (batch_size, input_size)
                    
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[:,t,:] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < config.teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:,t,:] = decoder_output
                                decoder_input = target_batch[:, t, :]

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:,t,:] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[:,t,:] = decoder_output

                            # predict with teacher forcing
                            if random.random() < config.teacher_forcing_ratio:
                                decoder_input = target_batch[:, t, :]

                            # predict recursively
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and config.teacher_forcing_ratio > 0:
                    config.teacher_forcing_ratio = config.teacher_forcing_ratio - 0.002

                if it % config.eval_freq == 0:
                    test_eval_dict = self.evaluate_batch(X_test=X_test, Y_test=Y_test)
                    train_eval_dict = self.evaluate_batch(X_test=X_train, Y_test=Y_train)

                    batch_test_loss = test_eval_dict["rmse"].item()
                    batch_train_loss = train_eval_dict["rmse"].item()
                    
                    test_rmse.append(batch_test_loss)
                    train_rmse.append(batch_train_loss)
                    
                    if early_stop and early_stopper.early_stop(batch_test_loss):
                        print("Early stopping")
                        break
                # progress bar
                metrics = {
                        "loss":batch_loss,
                        "test_rmse":batch_test_loss,
                        "train_rmse":batch_train_loss
                        }
                # tr.set_postfix(loss="{0:.3e}".format(batch_loss))
                tr.set_postfix(metrics)
                wandb.log(metrics)
        
        test_eval_dict = self.evaluate_batch(X_test=X_test, Y_test=Y_test)
        train_eval_dict = self.evaluate_batch(X_test=X_train, Y_test=Y_train)
        wandb.summary['test_rmse'] = test_eval_dict["rmse"].item()
        wandb.summary['train_rmse'] = train_eval_dict["rmse"].item()
        wandb.finish()
        
        return losses, test_rmse, train_rmse

    
    def predict_batch(self, input_tensor, target_len):
        '''
        : param input_tensor:      input data (batch, seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict (30)
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''
        batch_size = input_tensor.shape[0]
        
        encoder_output, encoder_hidden = self.encoder(input_tensor)
        
        outputs = torch.zeros(batch_size, target_len, self.output_size, device=self.device)  # input_tensor.shape[2])
        
        decoder_input = torch.zeros(batch_size, self.output_size, device=self.device)  # input_tensor[-1, :, :]%%!
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:,t,:] = decoder_output
            decoder_input = decoder_output
    
        np_outputs = outputs.detach()
        return np_outputs
    
    def evaluate_batch(self, X_test=None, Y_test=None, unnorm=True):
    
        y_pred = self.predict_batch(X_test, self.utils.output_window)
        
        if unnorm:
            # unnormalize the data
            y_pred = y_pred*self.utils.y_std + self.utils.y_mean
            Y_test = Y_test*self.utils.y_std + self.utils.y_mean
        
        rmse = (((y_pred-Y_test)**2).mean())**0.5
        
        evaluate_dict = {
            "y_pred":y_pred,
            "y_true":Y_test,
            "rmse":rmse,
        }
        return evaluate_dict

    def plot_err_win(self, X_test=None, Y_test=None, unnorm=True):
    
        y_pred = self.predict_batch(X_test, self.utils.output_window)
        
        if unnorm:
            y_pred = y_pred*self.utils.y_std + self.utils.y_mean
            Y_test = Y_test*self.utils.y_std + self.utils.y_mean
        
        rmse = (((y_pred-Y_test)**2).mean())**0.5
        
        err_vs_ws = []
    
        for ws in range(1, self.utils.output_window+1):
            #err_vs_ws.append((((y_pred[:, :ws, :]-Y_test[:, :ws, :])**2).mean())**0.5)
            err_vs_ws.append(((((y_pred[:, :ws, :]-Y_test[:, :ws, :])**2).mean(axis=1))**0.5).mean())
        
        plt.figure(figsize=(20,4), dpi=150)
        plt.grid("on", alpha=0.5)
        plt.plot(list(range(1,self.utils.output_window+1)), [i.cpu().numpy() for i in err_vs_ws])
        plt.xlabel("Window size")
        plt.ylabel("RMSE")
        plt.show()