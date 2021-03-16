import argparse
import socket                
import random
import numpy as np 
import threading
import time
import math
import os
from skopt import Optimizer
from skopt.space import Real
from scipy.interpolate import UnivariateSpline
from scipy import optimize 
import sys
import logging

logging.basicConfig(level=logging.INFO)

def f(b, x, y):
    A = np.vstack([np.exp(-np.exp(b) * x), np.ones(len(x))]).T
    res = np.linalg.lstsq(A, y, rcond=None)[1]
    return res

def spline_iter(xs, ys, is_training, spline_deg=2, filter_ratio=0.03, num_of_iter=10, bound=0.5):
    """ Use iterative spline to eliminate noise and outliers in the loss series.
        is_training specifies whether the loss series in use is training loss or validation loss.
    """
    bound = xs[int((len(xs) - 1) * bound)]
    if is_training:
        num_of_iter = 10
    else:
        num_of_iter = 1

    for _ in range(num_of_iter):
        spline_ys = UnivariateSpline(xs, ys, k=spline_deg)(xs)
        dys = np.abs(ys - spline_ys)

        if is_training:
            outliers = set(sorted(range(len(dys)), key=lambda i: dys[i])[int(round(-len(dys) * filter_ratio)):])
        else:
            outliers = set(sorted(range(len(dys)), key=lambda i: dys[i])[-1:])
        outliers = [i for i in outliers if i < bound]

        xs2 = np.zeros(len(xs) - len(outliers))
        ys2 = np.zeros(len(xs) - len(outliers))
        i1 = 0
        for i2 in range(len(xs)):
            if i2 not in outliers:
                xs2[i1], ys2[i1] = xs[i2], ys[i2]
                i1 += 1
        xs, ys = xs2, ys2
    return xs, ys

def exp_forecast(loss_series, end_step, is_training, spline_order=2):
    """ Do exponential forecasting on a loss series."""
    xs = np.arange(end_step - len(loss_series), end_step)
    xs2, ys2 = spline_iter(xs, loss_series, is_training)
    ys = UnivariateSpline(xs2, ys2, k=spline_order)(xs)
    logging.debug('ys after spline iter: {}'.format(ys))
    b = optimize.fmin(f, 0, args=(xs, ys), xtol=1e-5, ftol=1e-5, disp=False)[0]
    b = -np.exp(b)
    A = np.vstack([np.exp(b * xs), np.ones(len(xs))]).T
    a, c = np.linalg.lstsq(A, ys, rcond=None)[0]
    return a, b, c

class RingBuffer:
    """ A class for storing and manipulating loss series and do exponential forecasting. """

    def __init__(self, size):
        self.data = [None for i in range(size)]

    def reset(self):
        self.data = [None for i in self.data]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

    def average(self):
        return sum(self.data)/len(self.data)

    def exponential_forcast(self, pred_index, is_training):
        loss_series = self.data[:]
        end_epoch = len(loss_series)
        x = np.arange(end_epoch - len(loss_series), end_epoch)
        y = np.array(loss_series)
        a3, b3, c3 = exp_forecast(y, len(y), is_training)
        forcast_y = a3 * np.exp(b3 * pred_index) + c3
        logging.debug("Exponential fit: {}, {}, {}".format(a3, b3, c3))
        return forcast_y

class Controller(object):
    def __init__(self, host, port, min_lr, max_lr):
        # Constants
        EXPLOITATION_STEP = 1000
        LR_STEPS = 100
        RING_BUFFER_LEN = 100
        LR_TO_EXPLORE = 10
        TAU_MAX = 8000

        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.host = host
        self.port = port
        self.threads = []
        self.num_threads = 1
        self.event = threading.Event()
        self.sock = socket.socket()
        self.sock.bind((self.host, self.port))
        self.global_step = 0
        self.last_total_loss = 0.0
        self.average_loss = 0.0
        self.loss_vector = []
        self.lr = 0
        self.lr_steps = LR_STEPS 
        self.tau_max = TAU_MAX
        self.val_freq = int(self.lr_steps/16)
        self.lr_counter = 0
        self.lr_to_explore = LR_TO_EXPLORE
        self.BO_stage = True
        self.val_stage = False
        self.message = ''
        self.loss_after_exploitation = None
        self.ring_buffer_len = RING_BUFFER_LEN
        if self.val_stage:
            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len // self.val_freq)
        else:
            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)

        self.exploitation_step = EXPLOITATION_STEP
        self.exploitation_flag = False 
        self.exploitation_counter = 0
        
        self.opt = None
        self.x_func_dict = dict()
        self.x_iters = []
        self.func_val_iters = []

        self.num_ranks = 0
        self.finished_minions = 0
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()

    def listen(self):
        self.sock.listen(20)
        while True:
            client, address = self.sock.accept()
            logging.info('[Server]: Got connection from {}'.format(address))
            self.threads.append(threading.Thread(target = self.run, args = (client, address, self.event)))
            if len(self.threads) == self.num_threads:
                self.num_ranks = len(self.threads)
                self.num_minions = self.num_ranks - 1
                logging.info('[Server]: num_ranks: {}'.format(self.num_ranks))
                for thread in self.threads:
                    thread.start()
                logging.info('[Server]: threads started')
                for thread in self.threads:
                    thread.join()
                    self.threads = []
                logging.info('[Server]: join finished')
                sys.exit()

    def run(self, c, address, event):
        size = 1024
        while True:
            data = c.recv(size).decode()
            if not data:
                sys.exit()
            logging.debug(data.split(','))
            total_loss = float(data.split(',')[-1])
            self.lock2.acquire()
            self.loss_vector.append(total_loss)

            # compute average loss across ranks
            if len(self.loss_vector) == self.num_ranks:
                self.average_loss = sum(self.loss_vector) / len(self.loss_vector) 
                logging.info('[Server]: average loss = {}, step = {}'.format(self.average_loss, self.global_step))
                if self.val_stage:
                    if 'val' in data:
                        self.ring_loss_buffer.append(self.average_loss)
                    else:
                        self.global_step += 1
                else:
                    self.ring_loss_buffer.append(self.average_loss)
                    self.global_step += 1
                self.loss_vector = []
            self.lock2.release()

            if 'val' in data:
                c.send(str(self.lr).encode('utf-8')) 
                continue

            if 'minion' in data:
                # blocking
                event_is_set = event.wait()
                c.send(self.message.encode('utf-8')) 
                logging.debug('[Server] message: %s', self.message)
                self.lock1.acquire()
                self.finished_minions += 1
                self.lock1.release()
                if self.finished_minions == self.num_minions:
                    event.clear()
                    self.finished_minions = 0
                continue

            else:
                if data.startswith('startBO'):
                    self.last_total_loss = self.average_loss
                    self.init_loss = self.average_loss
                    self.loss_after_exploitation = self.average_loss

                # exploitation stage -- actual training stage using the best-found LR
                if self.exploitation_flag:
                    logging.debug('[Server exploitation]: average loss ' + str(self.average_loss) + ' step=' + str(self.global_step))
                    if self.exploitation_counter == self.exploitation_step:
                        self.BO_stage = True
                        self.exploitation_flag = False
                        self.exploitation_counter = 0
                        logging.info('[Server]: exploitation stage done')
                        logging.info('[Server]: reconfigure...')
                        if self.lr_steps < self.tau_max / 10:
                            self.lr_steps = self.lr_steps * 2
                            self.val_freq = int(self.lr_steps/16)
                            self.ring_buffer_len = self.lr_steps 
                            self.exploitation_step = self.exploitation_step * 2
                            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)
                        else:
                            self.val_stage = True
                        if self.val_stage:
                            self.ring_loss_buffer = RingBuffer(self.ring_buffer_len // self.val_freq)
                        self.loss_after_exploitation = self.average_loss

                        self.message = 'save'
                        c.send(self.message.encode('utf-8'))
                        event.set()
                        continue

                    else:
                        self.exploitation_counter += 1
                        self.message = str(self.lr)
                        c.send(str(self.lr).encode('utf-8')) 
                        event.set()
                        continue

                # BO stage -- LR search stage
                if self.BO_stage:
                    self.opt = Optimizer([Real(self.min_lr, self.max_lr, 'log-uniform')], "GP", n_initial_points=1, acq_func='LCB', acq_func_kwargs={'kappa':1e6})
                    self.BO_stage = False
                    self.lr = self.opt.ask()[0]

                    # prevent BO in scikit-optimize from searching for the same LR explored before
                    while True:
                        if self.lr in self.x_func_dict:
                            self.opt.tell([self.lr], self.x_func_dict[self.lr])
                            self.lr = self.opt.ask()[0]
                        else:
                            break

                    self.message = str(','.join(('ckpt', str(self.lr))))
                    c.send(','.join(('ckpt', str(self.lr))).encode('utf-8'))
                    event.set()
                    logging.debug('[Server]: checkpoint command sent')
                    continue

                # ask BO to suggest the next LR 
                if self.lr_counter == self.lr_steps:
                    logging.debug('ring_buffer: {}'.format(self.ring_loss_buffer.get()))
                    if any([math.isnan(x) for x in self.ring_loss_buffer.get()]):
                        predicted_loss = "nan"
                    elif self.val_stage:
                        predicted_loss = self.ring_loss_buffer.exponential_forcast(pred_index=int(self.exploitation_step/self.val_freq), is_training=False)
                        current_loss = sum(self.ring_loss_buffer.get()[-1:])/1.0
                    else:
                        predicted_loss = self.ring_loss_buffer.exponential_forcast(pred_index=self.exploitation_step, is_training=True)
                        current_loss = sum(self.ring_loss_buffer.get()[-10:])/10.0

                    logging.info('[Server]: predicted loss: {} due to LR {}'.format(predicted_loss, self.lr))

                    # Huge loss jump can make the exponential prediction inaccurate, so set a threshold here. 
                    #if self.loss_after_exploitation is not None and max(self.ring_loss_buffer.get()) > 10 * self.loss_after_exploitation:
                    #    predicted_loss = current_loss 
                    #    logging.info('New predicted_loss: ' + str(predicted_loss))

                    if self.loss_after_exploitation is not None and max(self.ring_loss_buffer.get()) >= 1.0 * self.init_loss and self.val_stage:
                        predicted_loss = current_loss 
                        logging.info('[Server]: New predicted_loss: ' + str(predicted_loss))

                    if self.val_stage:
                        self.ring_loss_buffer = RingBuffer(int(math.floor(self.ring_buffer_len)/self.val_freq))
                    else:
                        self.ring_loss_buffer = RingBuffer(self.ring_buffer_len)

                    # feed a (LR, predicted loss in tau steps) instance to BO.
                    if str(predicted_loss) == 'nan':
                        self.opt.tell([float(self.lr)], 1e6)
                    else:
                        self.opt.tell([float(self.lr)], predicted_loss)
                    self.x_iters.append(float(self.lr))
                    self.func_val_iters.append(predicted_loss)
                    self.x_func_dict[self.lr] = predicted_loss
                    self.lr_counter = 1

                    if len(self.func_val_iters) == self.lr_to_explore:
                        min_index = self.func_val_iters.index(min(self.func_val_iters))

                        # log the best lr found for the next stage.
                        logging.info('[Server]: best LR: {}, min loss: {}'.format(self.x_iters[min_index], self.func_val_iters[min_index]))

                        self.lr = self.x_iters[min_index]
                        self.message = str(','.join(('restore', str(self.lr))))
                        c.send(','.join(('restore', str(self.lr))).encode('utf-8'))
                        event.set()
                        logging.debug('[Server]: restore command sent')
                        self.exploitation_flag = True

                        self.func_val_iters = []
                        self.x_iters = []
                        self.x_func_dict = dict()
                    else:
                        # Ask BO for the next LR to explore
                        self.lr = self.opt.ask()[0]
                        while True:
                            if self.lr in self.x_func_dict:
                                self.opt.tell([self.lr], self.x_func_dict[self.lr])
                                self.lr = self.opt.ask()[0]
                            else:
                                break
                        self.message = str(','.join(('restore', str(self.lr))))
                        c.send(','.join(('restore', str(self.lr))).encode('utf-8'))
                        event.set()
                        logging.debug('[Server]: restore command sent')
                else:
                    self.lr_counter += 1
                    if self.val_stage and self.lr_counter % self.val_freq == 0:
                        self.message = "evaluate"
                        c.send(self.message.encode('utf-8')) 
                    else:
                        self.message = str(self.lr)
                        c.send(str(self.lr).encode('utf-8')) 
                    event.set()

def main():
    parser = argparse.ArgumentParser(description='AutoLRS server.')
    parser.add_argument('--min_lr', 
                        help='minimum LR',
                        required=True)
    parser.add_argument('--max_lr', 
                        help='maximum LR',
                        required=True)
    parser.add_argument('--host', 
                        help='host',
                        default='localhost',
                        type=str)
    parser.add_argument('--port', 
                        help='port',
                        required=True,
                        type=int)
    args = parser.parse_args()
    Controller(args.host, args.port, args.min_lr, args.max_lr).listen()

if __name__ == '__main__': 
    main() 

