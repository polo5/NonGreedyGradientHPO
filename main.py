"""
This uses forward mode differentiation to meta learn the
learning rate, momentum, and weights decay schedules.
There is no greediness assumption here,
i.e. the entire run makes up the inner loop.
"""

import os
import time
import shutil
import torch
import torch.optim as optim
import pickle

from utils.logger import *
from utils.helpers import *
from utils.datasets import *
from models.selector import *


class MetaLearner(object):
    def __init__(self, args):
        self.args = args

        ## Optimization
        self.hypers_init()
        self.outer_optimizer_init()
        self.cross_entropy = nn.CrossEntropyLoss()

        ## Experiment Set Up
        self.best_run = 0
        self.best_validation_acc = 0
        ns, learnables = (self.args.n_lrs, self.args.n_moms, self.args.n_wds), (self.args.learn_lr, self.args.learn_mom, self.args.learn_wd)
        self.all_lr_schedules, self.all_mom_schedules, self.all_wd_schedules  = [torch.zeros((self.args.n_runs+1, n)) for n in ns] #+1 since save init schedules and last schedule
        self.all_lr_raw_grads, self.all_mom_raw_grads, self.all_wd_raw_grads = [torch.zeros((self.args.n_runs, n)) if l else None for (n,l) in zip(ns, learnables)]
        self.all_lr_smooth_grads, self.all_mom_smooth_grads, self.all_wd_smooth_grads = [torch.zeros((self.args.n_runs, n)) if l else None for (n,l) in zip(ns, learnables)]

        self.experiment_path = os.path.join(self.args.log_directory_path, self.args.experiment_name)
        self.checkpoint_path = os.path.join(self.experiment_path, 'checkpoint.pth.tar')
        if os.path.exists(self.experiment_path):
            if self.args.use_gpu:
                if os.path.isfile(self.checkpoint_path):
                    raise NotImplementedError("Experiment folder exists. TODO: code to restart from checkpoint")
            else:
                shutil.rmtree(self.experiment_path) # clear debug logs on cpu
                os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)
        copy_file(os.path.realpath(__file__), self.experiment_path) # save this python file in logs folder

        ## Save and Print Args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')
        print('\nLogging every {} runs and every {} epochs per run\n'.format(self.args.run_log_freq, self.args.epoch_log_freq))

    def hypers_init(self):
        """ initialize hyperparameters """

        lrs = self.args.inner_lr_init*torch.ones(self.args.n_lrs)
        self.inner_lrs = torch.tensor(lrs, requires_grad=self.args.learn_lr, device=self.args.device)

        moms = self.args.inner_mom_init*torch.ones(self.args.n_moms)
        self.inner_moms = torch.tensor(moms, requires_grad=self.args.learn_mom, device=self.args.device)

        wds = self.args.inner_wd_init*torch.ones(self.args.n_wds)
        self.inner_wds = torch.tensor(wds, requires_grad=self.args.learn_wd, device=self.args.device)

    def outer_optimizer_init(self):
        """ define which inner variables to meta learn """
        to_optimize = []
        if self.args.learn_lr: to_optimize.append(self.inner_lrs)
        if self.args.learn_mom: to_optimize.append(self.inner_moms)
        if self.args.learn_wd: to_optimize.append(self.inner_wds)
        self.outer_optimizer = optim.SGD(to_optimize, lr=self.args.outer_lr_init, momentum=self.args.outer_momentum)

    def get_hypers(self, epoch):
        """return hyperparameters to be used for given epoch"""
        lr_index = int(epoch * self.args.n_lrs / self.args.n_epochs_per_run)
        lr = float(self.inner_lrs[lr_index])

        mom_index = int(epoch * self.args.n_moms / self.args.n_epochs_per_run)
        mom = float(self.inner_moms[mom_index])

        wd_index = int(epoch * self.args.n_wds / self.args.n_epochs_per_run)
        wd = float(self.inner_wds[wd_index])

        return lr, mom, wd, lr_index, mom_index, wd_index

    def log_epoch_init(self):
        """init moving average metrics to log over one epoch"""
        self.running_train_loss, self.running_train_acc = AggregateTensor(), AggregateTensor()
        self.running_val_loss, self.running_val_acc = AggregateTensor(), AggregateTensor()
        if self.args.use_gpu: torch.cuda.reset_max_memory_allocated(device=None) #reset max gpu memory seen

    def log_epoch(self, epoch):
        """
        Log relevant metrics for given epoch to tensorboard and csv files.
        Note that one epoch may not contain all of the training images because
        some are taken out for validation
        """
        avg_test_loss, avg_test_acc = self.test(self.weights, fraction=1 if epoch==self.args.n_epochs_per_run-1 else 0.1) # evaluate 10% of test data to save time
        avg_val_loss, avg_val_acc = self.validate(self.weights, fraction=1 if epoch==self.args.n_epochs_per_run-1 else 0.5)
        avg_train_loss, avg_train_acc = float(self.running_train_loss.avg()), float(self.running_train_acc.avg())
        lr, mom, wd, lr_index, mom_index, wd_index = self.get_hypers(epoch)

        print('   epoch {}/{} --- Train Acc: {:02.2f}% -- Val Acc: {:02.2f}% -- Test acc: {:02.2f}%'.format(epoch+1, self.args.n_epochs_per_run, avg_train_acc*100, avg_val_acc*100, avg_test_acc*100))

        self.logger.scalar_summary('TRAIN/inner_lr', lr, epoch)
        self.logger.scalar_summary('TRAIN/inner_mom', mom, epoch)
        self.logger.scalar_summary('TRAIN/inner_wd', wd, epoch)
        self.logger.scalar_summary('TRAIN/loss', avg_train_loss, epoch)
        self.logger.scalar_summary('TRAIN/acc', avg_train_acc*100, epoch)
        self.logger.scalar_summary('VAL/loss', avg_val_loss, epoch)
        self.logger.scalar_summary('VAL/acc', avg_val_acc*100, epoch)
        self.logger.scalar_summary('TEST/loss', avg_test_loss, epoch)
        self.logger.scalar_summary('TEST/acc', avg_test_acc*100, epoch)
        if self.args.use_gpu: self.logger.scalar_summary('TRAIN/GPU_usage_MB', float(torch.cuda.max_memory_allocated())/1e6, epoch)
        self.logger.write_to_csv('log.csv')
        self.logger.writer.flush()

        self.log_epoch_init()

    def clip_hypers(self):
        """
        Clip hyperparameters.
        Typically isn't necessary when outer optimizer is stable.
        Note that algorithm should work well without this otherwise it means
        we need prior information on the correct range for each hyperparameter
        """
        with torch.no_grad():
            if self.args.learn_lr: self.inner_lrs.data = self.inner_lrs.data.clamp(*self.args.lr_clamp_range)
            if self.args.learn_mom: self.inner_moms.data = self.inner_moms.data.clamp(*self.args.mom_clamp_range)
            if self.args.learn_wd: self.inner_wds.data = self.inner_wds.data.clamp(*self.args.wd_clamp_range)

    def clip_hypergrads(self):
        """
        Tricky: sometimes we get very small hyper grads but they are very high quality
        since computed over lots of epochs. Sometimes we get super large hypergrads
        but they suck since some step exploded. So we can't just 'clip' hypergrads 
        within a fixed range.  
        :return: 
        """
        with torch.no_grad():

            if self.args.learn_lr:
                self.inner_lrs.grad[self.inner_lrs.grad != self.inner_lrs.grad] = 0  # replace nans with zeros
                for idx, (lr, lr_grad) in enumerate(zip(self.inner_lrs, self.inner_lrs.grad)):
                    if lr_grad < 0: #lr will increase to no more than (1+frac)*lr
                        delta = max(self.args.lr_max_change_thresh, lr*args.lr_max_percentage_change)
                        grad_bound = -delta/self.args.outer_lr_init
                        if lr_grad < grad_bound:
                            self.inner_lrs.grad[idx] = grad_bound
                    else: 
                        delta = max(self.args.lr_max_change_thresh, lr*args.lr_max_percentage_change)
                        grad_bound = delta/self.args.outer_lr_init
                        if lr_grad > grad_bound:
                            self.inner_lrs.grad[idx] = grad_bound

            if self.args.learn_mom:
                self.inner_moms.grad[self.inner_moms.grad != self.inner_moms.grad] = 0
                for idx, (mom, mom_grad) in enumerate(zip(self.inner_moms, self.inner_moms.grad)):
                    if mom_grad < 0: #lr will increase to no more than (1+frac)*lr
                        delta = max(self.args.mom_max_change_thresh, mom*args.mom_max_percentage_change)
                        grad_bound = -delta/self.args.outer_lr_init
                        if mom_grad < grad_bound:
                            self.inner_moms.grad[idx] = grad_bound
                    else: 
                        delta = max(self.args.mom_max_change_thresh, mom*args.mom_max_percentage_change)
                        grad_bound = delta/self.args.outer_lr_init
                        if mom_grad > grad_bound:
                            self.inner_moms.grad[idx] = grad_bound

            if self.args.learn_wd:
                self.inner_wds.grad[self.inner_wds.grad != self.inner_wds.grad] = 0
                for idx, (wd, wd_grad) in enumerate(zip(self.inner_wds, self.inner_wds.grad)):
                    if wd_grad < 0: #lr will increase to no more than (1+frac)*lr
                        delta = max(self.args.wd_max_change_thresh, wd*args.wd_max_percentage_change)
                        grad_bound = -delta/self.args.outer_lr_init
                        if wd_grad < grad_bound:
                            self.inner_wds.grad[idx] = grad_bound
                    else: 
                        delta = max(self.args.wd_max_change_thresh, wd*args.wd_max_percentage_change)
                        grad_bound = delta/self.args.outer_lr_init
                        if wd_grad > grad_bound:
                            self.inner_wds.grad[idx] = grad_bound

    def inner_loop(self):

        ## Network
        self.classifier = select_model(True, self.args.dataset, self.args.architecture,
                                       self.args.init_type, self.args.init_param,
                                       self.args.device).to(self.args.device)
        self.classifier.train()
        self.weights = self.classifier.get_param()
        velocity = torch.zeros(self.weights.numel(), requires_grad=False, device=self.args.device)

        ## Forward Mode Init
        if self.args.learn_lr:
            Z_lr = torch.zeros((self.weights.numel(), self.args.n_lrs), device=self.args.device)
            C_lr = torch.zeros((self.weights.numel(), self.args.n_lrs), device=self.args.device)
        else:
            Z_lr = None

        if self.args.learn_mom:
            Z_mom = torch.zeros((self.weights.numel(), self.args.n_moms), device=self.args.device)
            C_mom = torch.zeros((self.weights.numel(), self.args.n_moms), device=self.args.device)
        else:
            Z_mom = None

        if self.args.learn_wd:
            Z_wd = torch.zeros((self.weights.numel(), self.args.n_wds), device=self.args.device)
            C_wd = torch.zeros((self.weights.numel(), self.args.n_wds), device=self.args.device)
        else:
            Z_wd = None


        ## Inner Loop Over All Epochs
        for epoch in range(self.args.n_epochs_per_run):
            t0_epoch = time.time()
            log_this_epoch = self.log_this_run and (epoch % self.args.epoch_log_freq == 0 or epoch == self.args.n_epochs_per_run - 1)
            lr, mom, wd, lr_index, mom_index, wd_index = self.get_hypers(epoch)

            for batch_idx, (x_train, y_train) in enumerate(self.train_loader):
                #t0_batch = time.time()
                x_train, y_train = x_train.to(device=self.args.device), y_train.to(device=self.args.device)
                train_logits = self.classifier.forward_with_param(x_train, self.weights)
                train_loss = self.cross_entropy(train_logits, y_train)
                train_acc = accuracy(train_logits.data, y_train, topk=(1,))[0]
                if log_this_epoch:
                    self.running_train_loss.update(train_loss)
                    self.running_train_acc.update(train_acc)

                grads = torch.autograd.grad(train_loss, self.weights, create_graph=True)[0]
                if self.args.clamp_grads: grads.clamp_(-self.args.clamp_grads_range, self.args.clamp_grads_range)

                if self.args.learn_lr:
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_lrs),device=self.args.device)
                    for j in range(lr_index + 1):
                        retain = (j != lr_index) or self.args.learn_mom or self.args.learn_wd
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_lr[:, j], self.weights, retain_graph=retain)[0]
                    
                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = Z_lr*(1 - lr*wd) - lr*H_times_Z
                    B = - mom*lr*C_lr
                    B[:,lr_index] -= grads.detach() + wd*self.weights.detach() + mom*velocity
                    C_lr = mom*C_lr + H_times_Z + wd*Z_lr

                    Z_lr = A_times_Z + B


                if self.args.learn_mom:
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_moms),device=self.args.device)
                    for j in range(mom_index + 1):
                        retain = (j != mom_index) or self.args.learn_wd
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_mom[:, j], self.weights, retain_graph=retain)[0]

                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = (1 - lr*wd)*Z_mom - lr*H_times_Z
                    B = -lr*mom*C_mom
                    B[:, mom_index] -= lr*velocity
                    C_mom = mom*C_mom + H_times_Z + wd * Z_mom
                    C_mom[:, mom_index] += velocity

                    Z_mom = A_times_Z + B

                if self.args.learn_wd:
                    H_times_Z = torch.zeros((self.weights.numel(), self.args.n_wds),device=self.args.device)
                    for j in range(wd_index + 1):
                        retain = (j != wd_index)
                        H_times_Z[:, j] = torch.autograd.grad(grads @ Z_wd[:, j], self.weights, retain_graph=retain)[0]
                    
                    if self.args.clamp_HZ: H_times_Z.clamp_(-self.args.clamp_HZ_range, self.args.clamp_HZ_range)
                    A_times_Z = (1 - lr*wd)*Z_wd - lr*H_times_Z
                    B = - lr*mom*C_wd
                    B[:, wd_index] -= lr*self.weights.detach()
                    C_wd = mom*C_wd + H_times_Z + wd*Z_wd
                    C_wd[:, wd_index] += self.weights.detach()

                    Z_wd = A_times_Z + B

                ## SGD update
                self.weights.detach_(), grads.detach_()
                velocity = velocity*mom + (grads + wd*self.weights)
                self.weights = self.weights - lr*velocity
                self.weights.requires_grad_()

            print(f'--- Ran epoch {epoch} in {format_time(time.time()-t0_epoch)} ---')
            if log_this_epoch: self.log_epoch(epoch)


        return Z_lr, Z_mom, Z_wd

    def outer_loop(self, run_idx, Z_lr_final, Z_mom_final, Z_wd_final):

        ## Calculate validation gradients with final weights of inner loop
        self.running_val_grad = AggregateTensor()
        for batch_idx, (x_val, y_val) in enumerate(self.val_loader): #need as big batches as train mode for BN train mode
            x_val, y_val = x_val.to(device=self.args.device), y_val.to(device=self.args.device)
            val_logits = self.classifier.forward_with_param(x_val, self.weights)
            val_loss = self.cross_entropy(val_logits, y_val)
            dLval_dw = torch.autograd.grad(val_loss, self.weights)[0]
            self.running_val_grad.update(dLval_dw)

        ## Calculate hypergrads
        print('')
        if self.args.learn_lr:
            self.inner_lrs.grad = self.running_val_grad.avg() @ Z_lr_final / self.n_batches_per_lr
            self.all_lr_raw_grads[run_idx] = self.inner_lrs.grad.detach()
            print('RAW LR GRADS: ', [float(i) for i in self.inner_lrs.grad])
        
        if self.args.learn_mom:
            self.inner_moms.grad = self.running_val_grad.avg() @ Z_mom_final / self.n_batches_per_mom
            self.all_mom_raw_grads[run_idx] = self.inner_moms.grad.detach()
            print('RAW MOM GRADS: ', [float(i) for i in self.inner_moms.grad])
        
        if self.args.learn_wd:
            self.inner_wds.grad = self.running_val_grad.avg() @ Z_wd_final / self.n_batches_per_mom
            self.all_wd_raw_grads[run_idx] = self.inner_wds.grad.detach()
            print('RAW WD GRADS: ', [float(i) for i in self.inner_wds.grad])
        

        self.clip_hypergrads()

        if self.args.learn_lr:
            self.all_lr_smooth_grads[run_idx] = self.inner_lrs.grad.detach()
            print('SMOOTH LR GRADS: ', [float(i) for i in self.inner_lrs.grad])

        if self.args.learn_mom:
            self.all_mom_smooth_grads[run_idx] = self.inner_moms.grad.detach()
            print('SMOOTH MOM GRADS: ', [float(i) for i in self.inner_moms.grad])

        if self.args.learn_wd:
            self.all_wd_smooth_grads[run_idx] = self.inner_wds.grad.detach()
            print('SMOOTH WD GRADS: ', [float(i) for i in self.inner_wds.grad])

        ## Update Hypers
        self.outer_optimizer.step()
        self.clip_hypers()

    def run(self):

        for run_idx in range(self.args.n_runs):

            ## Logging
            self.log_this_run = (run_idx % self.args.run_log_freq == 0 or run_idx==self.args.n_runs-1)
            self.run_folder = os.path.join(self.experiment_path, 'run{}'.format(run_idx))
            if os.path.exists(self.run_folder): shutil.rmtree(self.run_folder)
            if self.log_this_run:
                os.makedirs(self.run_folder)
                self.logger = Logger(log_dir=self.run_folder)
                self.log_epoch_init()
            
            ## Set up
            print(f'\nRun {run_idx+1}/{self.args.n_runs} using:')
            print(f'lrs = {self.inner_lrs.tolist()}')
            print(f'moms = {self.inner_moms.tolist()}')
            print(f'wds = {self.inner_wds.tolist()}')
            self.all_lr_schedules[run_idx], self.all_mom_schedules[run_idx], self.all_wd_schedules[run_idx] = self.inner_lrs.detach(), self.inner_moms.detach(), self.inner_wds.detach()
            self.save_state(run_idx) # state and lrs saved correspond to those set at the beginning of the run


            ## New data for each run
            self.train_loader, self.val_loader, self.test_loader  = get_loaders(datasets_path=self.args.datasets_path,
                                                                                dataset=self.args.dataset,
                                                                                train_batch_size=self.args.train_batch_size,
                                                                                val_batch_size=self.args.val_batch_size,
                                                                                val_source='train',
                                                                                val_train_fraction=self.args.val_train_fraction,
                                                                                val_train_overlap=self.args.val_train_overlap,
                                                                                workers=self.args.workers,
                                                                                train_infinite=False,
                                                                                val_infinite=False)
            self.n_batches_per_lr = len(self.train_loader) * self.args.n_epochs_per_run / self.args.n_lrs
            self.n_batches_per_mom = len(self.train_loader) * self.args.n_epochs_per_run / self.args.n_moms
            self.n_batches_per_wd = len(self.train_loader) * self.args.n_epochs_per_run / self.args.n_wds
            print(self.n_batches_per_lr, self.n_batches_per_mom, self.n_batches_per_wd)

            ## Update Hypers
            Z_lr_final, Z_mom_final, Z_wd_final = self.inner_loop()
            self.outer_loop(run_idx, Z_lr_final, Z_mom_final, Z_wd_final)

            ## See if schedule used for this run led to best validation
            _, val_acc = self.validate(self.weights)
            if val_acc > self.best_validation_acc:
                self.best_validation_acc = val_acc
                self.best_run = run_idx
                print(f'Best validation acc at run idx {run_idx}')

        ## Logging Final Metrics
        self.all_lr_schedules[run_idx+1], self.all_mom_schedules[run_idx+1], self.all_wd_schedules[run_idx+1] = self.inner_lrs.detach(), self.inner_moms.detach(), self.inner_wds.detach() #last schedule was never trained on
        self.save_state(run_idx+1)
        avg_test_loss, avg_test_acc = self.test(self.weights)

        return avg_test_acc

    def validate(self, weights, fraction=1.0):
        """ fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(self.val_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier.forward_with_param(x, weights)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.val_loader):
                    break

        self.classifier.train()
        return float(running_loss.avg()), float(running_acc.avg())

    def test(self, weights, fraction=1.0):
        """ fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier.forward_with_param(x, weights)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.test_loader):
                    break

        self.classifier.train()
        return float(running_loss.avg()), float(running_acc.avg())

    def save_state(self, run_idx):

        torch.save({'args': self.args,
                    'run_idx': run_idx,
                    'best_run': self.best_run,
                    'best_validation_acc': self.best_validation_acc,
                    'outer_optimizer': self.outer_optimizer.state_dict(),
                    'all_lr_schedules': self.all_lr_schedules,
                    'all_lr_raw_grads': self.all_lr_raw_grads,
                    'all_lr_smooth_grads': self.all_lr_smooth_grads,
                    'all_mom_schedules': self.all_mom_schedules,
                    'all_mom_raw_grads': self.all_mom_raw_grads,
                    'all_mom_smooth_grads': self.all_mom_smooth_grads,
                    'all_wd_schedules': self.all_wd_schedules,
                    'all_wd_raw_grads': self.all_wd_raw_grads,
                    'all_wd_smooth_grads': self.all_wd_smooth_grads}, self.checkpoint_path)


class BaseLearner(object):
    """
    Train from scratch using whole training set + standard Pytorch architectures
    """
    def __init__(self, args, lr_schedule, mom_schedule, wd_schedule, log_name):
        self.args = args
        self.inner_lrs = lr_schedule
        self.inner_moms = mom_schedule
        self.inner_wds = wd_schedule

        ## Loaders
        self.args.val_source = 'test' # retrain on full train set from scratch
        self.train_loader, self.val_loader, self.test_loader  = get_loaders(datasets_path=self.args.datasets_path,
                                                                            dataset=self.args.dataset,
                                                                            train_batch_size=self.args.train_batch_size,
                                                                            val_batch_size=self.args.val_batch_size,
                                                                            val_source=self.args.val_source,
                                                                            val_train_fraction=self.args.val_train_fraction,
                                                                            val_train_overlap=self.args.val_train_overlap,
                                                                            workers=self.args.workers,
                                                                            train_infinite=False,
                                                                            val_infinite=False)

        ## Optimizer
        self.classifier = select_model(False, self.args.dataset, self.args.architecture, self.args.activation,
                               self.args.norm_type, self.args.norm_affine, self.args.noRes,
                               self.args.init_type, self.args.init_param, self.args.init_norm_weights).to(self.args.device)
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.0, momentum=0.0, weight_decay=0.0) #set hypers manually later
        self.cross_entropy = nn.CrossEntropyLoss()

        ### Set up
        self.experiment_path = os.path.join(args.log_directory_path, args.experiment_name)
        self.logger = Logger(log_dir=os.path.join(self.experiment_path, log_name))

    def log_init(self):
        self.running_train_loss, self.running_train_acc = AggregateTensor(), AggregateTensor()

    def log(self, epoch, avg_train_loss, avg_train_acc):
        avg_test_loss, avg_test_acc = self.test(fraction=0.1 if epoch!=self.args.retrain_n_epochs-1 else 1)
        print('Retrain epoch {}/{} --- Train Acc: {:02.2f}% -- Test Acc: {:02.2f}%'.format(epoch+1, self.args.n_epochs_per_run, avg_train_acc * 100, avg_test_acc * 100))
        self.logger.scalar_summary('RE-RUN/train_loss', avg_train_loss, epoch)
        self.logger.scalar_summary('RE-RUN/train_acc', avg_train_acc * 100, epoch)
        self.logger.scalar_summary('RE-RUN/test_loss', avg_test_loss, epoch)
        self.logger.scalar_summary('RE-RUN/test_acc', avg_test_acc * 100, epoch)
        if self.args.use_gpu: self.logger.scalar_summary('RE-RUN/GPU_usage_MB', float(torch.cuda.memory_allocated())/1e6, epoch)
        self.logger.write_to_csv('re-run_log.csv')
        self.logger.writer.flush()

        self.log_init()

    def get_hypers(self, epoch):
        # TODO: freeze init epochs for some hypers only?
        lr_index = int(epoch * self.args.n_lrs / self.args.retrain_n_epochs) #NB: e.g. int(199*5/200)=4
        lr = float(self.inner_lrs[lr_index])

        mom_index = int(epoch * self.args.n_moms / self.args.retrain_n_epochs)
        mom = float(self.inner_moms[mom_index])

        wd_index = int(epoch * self.args.n_wds / self.args.retrain_n_epochs)
        wd = float(self.inner_wds[wd_index])

        return lr, mom, wd, lr_index, mom_index, wd_index

    def set_hypers(self, epoch):
        lr, mom, wd, lr_index, mom_index, wd_index = self.get_hypers(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom
            param_group['weight_decay'] = wd

        print(f'Setting: lr={lr}, mom={mom}, wd={wd}')

    def run(self):

        for epoch in range(self.args.n_epochs_per_run):
            self.set_hypers(epoch)
            avg_train_loss, avg_train_acc = self.train()
            self.log(epoch, avg_train_loss, avg_train_acc)

        test_loss, test_acc = self.test()
        return test_acc

    def train(self):
        self.classifier.train()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        for idx, (x,y) in enumerate(self.train_loader):
            x, y = x.to(device=self.args.device), y.to(device=self.args.device)
            logits = self.classifier(x)
            loss = self.cross_entropy(input=logits, target=y)
            acc1 = accuracy(logits.data, y, topk=(1,))[0]
            running_loss.update(loss, x.shape[0])
            running_acc.update(acc1, x.shape[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return running_loss.avg(), running_acc.avg()

    def test(self, fraction=1.0):
        """ fraction allows trading accuracy for speed when logging many times"""
        self.classifier.eval()
        running_acc, running_loss = AggregateTensor(), AggregateTensor()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(device=self.args.device), y.to(device=self.args.device)
                logits = self.classifier(x)
                running_loss.update(self.cross_entropy(logits, y), x.shape[0])
                running_acc.update(accuracy(logits, y, topk=(1,))[0], x.shape[0])
                if fraction < 1 and (batch_idx + 1) >= fraction*len(self.test_loader):
                    break

        self.classifier.train()
        return running_loss.avg(), running_acc.avg()


# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________

def make_experiment_name(args):

    ## Main
    experiment_name = f'FSL_{args.dataset}_{args.architecture}_nr{args.n_runs}_nepr{args.n_epochs_per_run}'
    if args.learn_lr: experiment_name += f'_learn{args.n_lrs}lrs-p{args.lr_max_percentage_change}-t{args.lr_max_change_thresh}'
    if args.learn_mom: experiment_name += f'_learn{args.n_moms}moms-p{args.mom_max_percentage_change}-t{args.mom_max_change_thresh}'
    if args.learn_wd: experiment_name += f'_learn{args.n_wds}wds-p{args.wd_max_percentage_change}-t{args.wd_max_change_thresh}'


    ## inner/outer init
    experiment_name += f'_init{args.init_type}-{args.init_param}'
    experiment_name += f'_tbs{args.train_batch_size}'
    experiment_name += f'_ilr{args.inner_lr_init}_imom{args.inner_mom_init}_iwd{args.inner_wd_init}'
    experiment_name += f'_olr{args.outer_lr_init}_omom{args.outer_momentum}'

    ## optional params
    if args.clamp_HZ: experiment_name += f'_HZclamp{args.clamp_HZ_range}'
    if args.clamp_grads: experiment_name += f'_gradsclamp{args.clamp_grads_range}'

    experiment_name += f'_S{args.seed}'

    return experiment_name


def main(args):
    set_torch_seeds(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    meta_learner = MetaLearner(args)
    meta_test_acc = meta_learner.run()

    to_print = '\n\nMETA TEST ACC: {:02.2f}%'.format(meta_test_acc*100)
    file_name = "final_meta_test_acc_{:02.2f}".format(meta_test_acc*100)
    create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))

    if args.retrain_from_scratch:
        ## Fetch schedules
        best_idx = meta_learner.best_run
        final_lr_schedule, final_mom_schedule, final_wd_schedule = meta_learner.all_lr_schedules[-1], meta_learner.all_mom_schedules[-1], meta_learner.all_wd_schedules[-1]
        best_lr_schedule, best_mom_schedule, best_wd_schedule = meta_learner.all_lr_schedules[best_idx], meta_learner.all_mom_schedules[best_idx], meta_learner.all_wd_schedules[best_idx]
        del meta_learner

        ## Retrain Last
        print(f'\n\n\n---------- RETRAINING FROM SCRATCH WITH LAST SCHEDULE (idx {args.n_runs}) ----------')
        print(f'lrs = {final_lr_schedule.tolist()}')
        print(f'moms = {final_mom_schedule.tolist()}')
        print(f'wds = {final_wd_schedule.tolist()}')

        log_name = f'Rerun_last_run_idx_{args.n_runs}'
        base_learner = BaseLearner(args, final_lr_schedule, final_mom_schedule, final_wd_schedule, log_name)
        if args.use_gpu: torch.cuda.empty_cache()
        base_test_acc = base_learner.run()
        to_print += '\nRE-RUN LAST SCHEDULE TEST ACC: {:02.2f}%'.format(base_test_acc*100)
        file_name = "Rerun_last_test_acc_{:02.2f}".format(base_test_acc*100)
        create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))


        ## Retrain Best Val
        print(f'\n\n\n---------- RETRAINING FROM SCRATCH WITH BEST SCHEDULE (idx {best_idx}) ----------')
        print(f'lrs = {best_lr_schedule.tolist()}')
        print(f'moms = {best_mom_schedule.tolist()}')
        print(f'wds = {best_wd_schedule.tolist()}')

        log_name = f'Rerun_best_run_idx_{best_idx}'
        base_learner = BaseLearner(args, best_lr_schedule, best_mom_schedule, best_wd_schedule, log_name)
        if args.use_gpu: torch.cuda.empty_cache()
        base_test_acc = base_learner.run()
        to_print += '\nRE-RUN BEST SCHEDULE TEST ACC: {:02.2f}%'.format(base_test_acc*100)
        file_name = "Rerun_best_test_acc_{:02.2f}".format(base_test_acc*100)
        create_empty_file(os.path.join(args.log_directory_path, args.experiment_name, file_name))

    print(to_print)


if __name__ == "__main__":
    import argparse
    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to GreedyGrad')

    ## Main
    parser.add_argument('--learn_lr', type=str2bool, default=True)
    parser.add_argument('--learn_mom', type=str2bool, default=True)
    parser.add_argument('--learn_wd', type=str2bool, default=True)
    parser.add_argument('--n_lrs', type=int, default=2)
    parser.add_argument('--n_moms', type=int, default=2)
    parser.add_argument('--n_wds', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--n_runs', type=int, default=2, help='number of full training runs. n_runs/run_log_freq folders will be created')
    parser.add_argument('--n_epochs_per_run', type=int, default=2, help='number of epochs to run before updating all learning rates')

    ## Architecture
    parser.add_argument('--architecture', type=str, default='LeNet')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal', 'zero', 'default'], help='network initialization scheme')
    parser.add_argument('--init_param', type=float, default=1, help='network initialization param: gain, std, etc.')
    parser.add_argument('--init_norm_weights', type=float, default=1, help='init gammas of BN')

    ## Inner Loop
    parser.add_argument('--inner_lr_init', type=float, default=0.1, help='SGD inner learning rate init')
    parser.add_argument('--inner_mom_init', type=float, default=0.5, help='SGD inner momentum init')
    parser.add_argument('--inner_wd_init', type=float, default=5e-4, help='SGD inner weight decay init')
    parser.add_argument('--train_batch_size', type=int, default=512)

    ## Outer Loop
    parser.add_argument('--outer_lr_init', type=float, default=0.1, help='Learning rate of all meta parameters')
    parser.add_argument('--outer_momentum', type=float, default=0.1)
    parser.add_argument('--val_batch_size', type=int, default=500)
    parser.add_argument('--val_train_fraction', type=float, default=0.05, help='ignored unless val_source=train')
    parser.add_argument('--val_train_overlap', type=str2bool, default=False, help='if True and val_source=train, val images are also in train set')

    ## Hypers & Hypergrads
    parser.add_argument('--lr_max_percentage_change', type=float, default=0.1, help='at each iteration grads changed so that each hyper can only change by this fraction (ignoring outer momentum)')
    parser.add_argument('--mom_max_percentage_change', type=float, default=0.1)
    parser.add_argument('--wd_max_percentage_change', type=float, default=0.1)
    parser.add_argument('--lr_max_change_thresh', type=float, default=0.02, help='clipping by percentage can be problematic if hyper becomes very small so this allows hyper to change up to that much')
    parser.add_argument('--mom_max_change_thresh', type=float, default=0.05, help='clipping by percentage can be problematic if hyper becomes very small so this allows hyper to change up to that much')
    parser.add_argument('--wd_max_change_thresh', type=float, default=1e-4, help='clipping by percentage can be problematic if hyper becomes very small so this allows hyper to change up to that much')
    parser.add_argument('--lr_clamp_range', nargs='*', type=float, default=[0, 1])
    parser.add_argument('--mom_clamp_range', nargs='*', type=float, default=[0, 1])
    parser.add_argument('--wd_clamp_range', nargs='*', type=float, default=[0, 1])
    parser.add_argument('--clamp_HZ', type=str2bool, default=True)
    parser.add_argument('--clamp_HZ_range', type=float, default=100, help='clamp to +/- that')
    parser.add_argument('--clamp_grads', type=str2bool, default=True)
    parser.add_argument('--clamp_grads_range', type=float, default=3, help='clamp inner grads for each batch to +/- that')

    ## Other
    parser.add_argument('--retrain_from_scratch', type=str2bool, default=False, help='retrain from scratch with learned lr schedule')
    parser.add_argument('--retrain_n_epochs', type=int, default=4, help='interpolates from learned schedule, -1 for same as n_epochs_per_run')
    parser.add_argument('--datasets_path', type=str, default="/home/paul/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="./logs/")
    parser.add_argument('--epoch_log_freq', type=int, default=1, help='every how many epochs to save summaries')
    parser.add_argument('--run_log_freq', type=int, default=1, help='every how many runs to save the whole run')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    args = parser.parse_args()

    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    assert args.n_epochs_per_run % args.n_lrs == 0 #for simplicity each hyper must be used for same number of steps
    assert args.n_epochs_per_run % args.n_moms == 0
    assert args.n_epochs_per_run % args.n_wds == 0
    if args.retrain_n_epochs < 0: args.retrain_n_epochs = args.n_epochs_per_run

    args.experiment_name = make_experiment_name(args)

    print('\nRunning on device: {}'.format(args.device))
    if args.use_gpu: print(torch.cuda.get_device_name(0))


    main(args)


