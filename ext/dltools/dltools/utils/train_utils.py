import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod, abstractproperty
import os


__all__ = [
    'ConvergenceEarlyStopping',
    'MetricEarlyStopping'
    ]


class AbstractEarlyStopping(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractproperty
    def stop(self):
        pass


class ConvergenceEarlyStopping(AbstractEarlyStopping):

    '''
    A mechanism for pre-stopping training using pipe

    self.curr_metric_value - quality value on current epoch
    self.num_epochs - total number of training epochs in the hpo algorithm
    self.curr_epoch - current epoch
    self.history - list with history of metric value
    self.ema_history - list with history of ema
    self.save_path - path to save figure with time series and stop predict
    self.lyambda - the rate of forgetting the history of the series
    self.pre_stop_epochs - number of epochs before stopping, when stop criterion is True
    self.stop_epoch - None if the pre-stop mechanism has not been used, 
                      otherwise the number of the stop epoch
    self.stop_epoch_count - the number of consecutive epochs when stop criterion is True
    '''

    def __init__(self, num_epochs, save_path, pipe_width=0.001,
                 lyambda=0.25, pre_stop_epochs=15):
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.pipe_width = pipe_width
        self.curr_epoch = 0
        self.curr_metric_value = 0
        self.history = [0]
        self.ema_history = [0]
        self.lyambda = lyambda
        self.pre_stop_epochs = pre_stop_epochs
        self.stop_epoch = None
        self.stop_epoch_count = 0
        self.mean_value = 0

    def add(self, value):
        '''add metric value to history'''
        self.curr_metric_value = value
        self.curr_epoch += 1
        self.history.append(value)
        ema_value = self.__ema(value)
        self.ema_history.append(ema_value)
    
    @property
    def stop(self):
        return self.__stop()
    
    def __stop(self):

        '''calculate current stop-value'''

        if self.curr_epoch < self.pre_stop_epochs:
            return False
        
        candidates = self.ema_history[-self.pre_stop_epochs:]
        self.mean_value = sum(candidates) / self.pre_stop_epochs
        in_pipe = lambda x: self.mean_value - self.pipe_width <= x \
                         <= self.mean_value + self.pipe_width
        in_pipe_lst = [in_pipe(x) for x in candidates]

        if all(in_pipe_lst):
            self.stop_epoch = self.curr_epoch
            return True
        else:
            return False

    def __ema(self, value):
        '''calculate ema value'''
        return self.lyambda * value + (1-self.lyambda) * self.ema_history[-1]

    def __str__(self) -> str:
        s = [
            'ConvergenceEarlyStoppingObject:',
            f'pipe_width={self.pipe_width}',
            f'save_path={self.save_path}'
        ]
        return '\n'.join(s)

    def plot(self):

        '''plot a graph with pre-stopping point'''

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.history, color='black', label='real_metric_value')
        ax.plot(self.ema_history, color='red', label='ema_metric_value')
        ax.axhline(y=self.mean_value-self.pipe_width, color='gray', linestyle='--', label='pipe')
        ax.axhline(y=self.mean_value+self.pipe_width, color='gray', linestyle='--', label='pipe')
        ax.axvline(x=self.num_epochs, color='blue', label='last_epoch')
        
        ax.grid()
        ax.set_xlabel('epoch')
        ax.set_ylabel('metric_value')
        ax.legend()
        fig.savefig(os.path.join(self.save_path, 'early_stopping.png'))


class MetricEarlyStopping(AbstractEarlyStopping):

    '''
    A mechanism for pre-stopping training using target metric value

    self.target_metric_value - custom setting for the required quality value
    self.curr_metric_value - quality value on current epoch
    self.num_epochs - total number of training epochs in the hpo algorithm
    self.curr_epoch - current epoch
    self.history - list with history of metric value
    self.ema_history - list with history of ema
    self.save_path - path to save figure with time series and stop predict
    self.lyambda - the rate of forgetting the history of the series
    self.min_learn_epochs - min epochs to learn without using pre-stop
    self.pre_stop_epochs - number of epochs before stopping, when stop criterion is True
    self.stop_epoch - None if the pre-stop mechanism has not been user, 
                      otherwise the number of the stop epoch
    self.stop_epoch_count - the number of consecutive epochs when stop criterion is True
    self.reverse_tan - True if tan value should be negative, default False
    '''

    def __init__(self, target_metric_value, num_epochs, 
                 save_path, lyambda=0.25, min_learn_epochs=5,
                 pre_stop_epochs=10, reverse_tan=False):
        self.target_metric_value = target_metric_value
        self.curr_metric_value = 0
        self.num_epochs = num_epochs
        self.curr_epoch = 0
        self.history = [0]
        self.ema_history = [0]
        self.save_path = save_path
        self.lyambda = lyambda
        self.min_learn_epoch = min_learn_epochs
        self.pre_stop_epochs = pre_stop_epochs
        self.stop_epoch = None
        self.stop_epoch_count = 0
        self.reverse_tan=reverse_tan

    def add(self, value):
        '''add metric value to history and update ema'''
        self.curr_metric_value = value
        self.curr_epoch += 1
        self.history.append(value)
        ema_value = self.__ema(value)
        self.ema_history.append(ema_value)

    @property
    def stop(self):
        return self.__stop()

    def __stop(self) -> bool:

        '''calculate current stop-value'''

        if self.curr_epoch < self.min_learn_epoch:
            return False

        xa, xb, = self.curr_epoch - 1, self.curr_epoch
        ya, yb = self.ema_history[-2], self.ema_history[-1]
        x = (xb - xa)/(yb - ya) * (self.target_metric_value - ya) + xa

        dy = self.ema_history[-1] - self.ema_history[-2]
        tan_alpha = - 1 / dy if self.reverse_tan else 1 / dy
        
        if tan_alpha <= 0 or x > self.num_epochs:
            self.stop_epoch_count += 1
        else:
            self.stop_epoch_count = 0
        
        if self.stop_epoch_count >= self.pre_stop_epochs:
            self.stop_epoch = self.curr_epoch
            return True
        else:
            return False

    def __ema(self, value):
        '''calculate ema value'''
        return self.lyambda * value + (1-self.lyambda) * self.ema_history[-1]

    def __str__(self) -> str:
        s = [
            'MetricEarlyStoppingObject:',
            f'target_metric_value={self.target_metric_value}',
            f'num_epochs={self.num_epochs}',
            f'save_path={self.save_path}'
        ]
        return '\n'.join(s)

    def plot(self):

        '''plot a graph with pre-stopping point'''

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.history, color='black', label='real_metric_value')
        ax.plot(self.ema_history, color='red', label='ema_metric_value')
        ax.axhline(y=self.target_metric_value, color='gray', linestyle='--', label='target')
        ax.axvline(x=self.num_epochs, color='blue', label='last_epoch')

        if self.stop_epoch is not None:

            xa, xb, = self.curr_epoch - 1, self.curr_epoch
            ya, yb = self.ema_history[-2], self.ema_history[-1]
            x = (xb - xa)/(yb - ya) * (self.target_metric_value - ya) + xa

            ax.plot(
                [self.curr_epoch, x],
                [self.ema_history[-1], self.target_metric_value],
                'o-y', label='predicted'
            )
        
        ax.grid()
        ax.set_xlabel('epoch')
        ax.set_ylabel('metric_value')
        ax.legend()
        fig.savefig(os.path.join(self.save_path, 'early_stopping.png'))


if __name__ == '__main__':
    
    start_fn = lambda x: (np.log(x*(1 + (np.random.rand() - 0.5)*0.1)) + 3) * 0.1
    end_fn = lambda a, x: start_fn(a) * (1 - 0.1) + start_fn(x) * 0.1
    a = 40

    acc_series = [start_fn(x) if x <= a else end_fn(a, x) for x in np.arange(0.1, a+10, 0.1)]



    ### Usage example ###

    target_metric_value = 0.685 # user limit
    num_epochs = len(acc_series)
    save_path = os.path.abspath('')

    # es_agent = MetricEarlyStopping(
    #     target_metric_value=target_metric_value,
    #     num_epochs=num_epochs,
    #     save_path=save_path
    # )

    es_agent = ConvergenceEarlyStopping(
        num_epochs=num_epochs,
        save_path=save_path
    )

    # Train loop...
    for epoch in range(num_epochs):
        # ... train step ...
        current_acc = acc_series[epoch] # calculate current metrics
        es_agent.add(current_acc)
        print(es_agent.ema_history[-1], es_agent.history[-1])

        if es_agent.stop:
            print('Stopping training')
            break
    
    es_agent.plot()
