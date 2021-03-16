import torch
import os
import logging
import socket
import string
import random
import time

class AutoLRS():
    def __init__(self, model, optimizer, val_fn, listening_host='localhost', listening_port=12315, warmup_steps=0, warmup_lr=0, summary_steps=1):
        self._net = model
        self._optimizer = optimizer
        self._val_fn = val_fn 
        self._lr = 0.000001
        self._warmup_steps = warmup_steps
        self._warmup_lr = warmup_lr 
        self._global_step = 0
        self._socket = socket.socket()
        self._started = False
        self._summary_steps = summary_steps
        self._checkpoint_path = './checkpoint/autolrs_ckpt_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))  + '.pth'
        self._listening_host = listening_host
        self._listening_port = listening_port
        self._best_acc = 0

        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')

        self.connect_server()
    
    def connect_server(self):
        self._socket.connect((self._listening_host, self._listening_port))

    def _verbose_operation(self, _op):
        if self._global_step % self._summary_steps == 0:
            logging.info("[AutoLRS at {}] {}".format(self._global_step, _op))

    def save_variables(self):
        """Save model parameters and optimizer states."""
        _start_time = time.time()
        torch.save({
            'model_state_dict': self._net.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
            }, self._checkpoint_path)
        logging.info("[AutoLRS] backup variables, elapsed: {}s".format(time.time() - _start_time))

    def restore_variables(self):
        _start_time = time.time()
        checkpoint = torch.load(self._checkpoint_path)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("[AutoLRS] restoring variables, elapsed: {}s".format(time.time() - _start_time))

    def on_train_batch_end(self, loss):
        if self._global_step < self._warmup_steps:
            # linear warmup
            self._lr = (self._warmup_lr / self._warmup_steps) * (self._global_step + 1)
            for param_group in self._optimizer.param_groups:
            	param_group['lr'] = self._lr
            self._global_step += 1

        elif not self._started:
            self.save_variables()
            print("backup trainable variables to CPU") 
            self._started = True
            self._socket.send(",".join(('startBO', str(loss))).encode("utf-8"))
            self._verbose_operation("Start Bayesian Optimization(BO)")
            data = self._socket.recv(1024).decode("utf-8")
            self._verbose_operation("Received data: " + data)
            self._lr = (float(data.split(",")[-1]))
            for param_group in self._optimizer.param_groups:
            	param_group['lr'] = self._lr
        else:
            self._socket.send(','.join(('loss', str(loss))).encode('utf-8'))
            data = self._socket.recv(1024).decode("utf-8")
            self._verbose_operation("Received data: " + data)
            if data.startswith("restore"):
                self.restore_variables()
                self._verbose_operation("restore trainable variables")
            elif data.startswith("ckpt"):
                self.save_variables()
                self._verbose_operation("backup trainable variables")
            elif data.startswith('evaluate'):
                val_loss = self._val_fn()
                self._socket.send(",".join(("val_loss", str(val_loss))).encode("utf-8"))
                data = self._socket.recv(1024).decode("utf-8")
            elif data.startswith('save'):
                pass
            else:
                self._lr = (float(data.split(",")[-1]))
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = self._lr
                self._global_step += 1
