import datetime
import time
import torch
import tabulate as tb
import plot_utils
import numpy as np


class MetricLogger(object):
    def __init__(self, user_header=['train_acc', 'test_acc', 'train_loss', 'test_loss'],
                 lr_scheduler=None,
                 console_live_log=True, live_plot=['train_acc', 'test_acc']):
        self.console_live_log = console_live_log  # false if log is printed only at the end
        self.live_plot = live_plot  # true if drawing a live plot every epoch
        self.live_plot_data = None  # data for the live plot
        self.lr_scheduler = lr_scheduler
        self.user_header = user_header  # the user provided header
        self.header = ['epoch']  # build the actual header which has extra info
        self.header.extend(user_header)
        if lr_scheduler is not None:
            self.header.append('lr')
        self.header.extend(['runtime'])
        self.table = []  # the table containing all info ordered by header
        self.epoch = 0  # the current epoch
        self.runtime = 0  # the current runtime
        self.user_data = None  # the data of the current iteration provided by update(...)
        self.total_time = 0  # how long the total training took
        self.total_mem = 0  # total memory used

    def print_report(self):
        format_table(self.table, header=self.header, separators=False)
        print(f"\nTotal time: {str(datetime.timedelta(seconds=int(self.total_time)))}")
        print(f"Time per iteration: {str(datetime.timedelta(seconds=int(self.total_time / self.epoch)))}")
        print(f"Total mem: {self.total_mem}")

    def update(self, **kwargs):
        # get the data for the logs
        row = []
        # user_data = kwargs.items()
        for data_h in self.user_header:
            row.append(kwargs[data_h])
        self.user_data = row

        # get the data for the live_plot
        if self.live_plot is not None:
            row = []
            for data_h in self.live_plot:
                row.append(kwargs[data_h])
            self.live_plot_data = row

    def log(self, iterable):
        torch.cuda.reset_peak_memory_stats()
        if self.live_plot is not None:
            animator = plot_utils.LivePlotter(xlabel='epoch', ylabel='acc',
                                              legend=self.live_plot, xlim=[0, len(iterable) - 1])
        start = time.time()
        for obj in iterable:
            # save the current learning rate before it is updated
            if self.lr_scheduler is not None:
                last_lr = self.lr_scheduler.get_last_lr()[0]

            # run and benchmark the execution
            estart_time = time.time()
            yield obj
            runtime = time.time() - estart_time

            # produce the stats and append them to the table
            row = [self.epoch]
            row.extend(self.user_data)
            if self.lr_scheduler is not None:
                row.append(last_lr)
            self.epoch += 1
            self.runtime += runtime
            row.append(runtime)
            self.table.append(row)

            # log to console if live_log is on
            if self.console_live_log:
                eta_seconds = self.runtime / self.epoch * (len(iterable) - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                mem = int(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
                format_row_carriage_return(self.header + ['eta', 'mem'], row + [eta_string, mem])

            # update the plot if we are plotting live
            if self.live_plot is not None:
                animator.add(self.epoch - 1, self.live_plot_data)

        # benchmark over, track execution stats and print to console if requested
        self.total_time = time.time() - start
        self.total_mem = int(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if self.console_live_log:
            print('\r', end='')

    def report(self, best_epoch, best_acc):  # todo create another object which will contain this data
        return MetricReport(self.header, self.table, best_epoch, best_acc, self.total_time, self.total_mem, self.epoch)


class MetricReport:
    def __init__(self, header, table, best_epoch, best_acc, total_time, total_mem, epochs):
        self.header = header
        self.table = table
        self.best_epoch = best_epoch
        self.best_acc = best_acc
        self.total_time = total_time
        self.total_mem = total_mem
        self.epochs = epochs

    def print(self):
        format_table(self.table, header=self.header, separators=False)
        print(f"\nBest acc: {self.best_acc:3.2f}% @ epoch {self.best_epoch}")
        print(f"Total time: {str(datetime.timedelta(seconds=int(self.total_time)))}")
        print(f"Time per iteration: {str(datetime.timedelta(seconds=int(self.total_time / self.epochs)))}")
        print(f"Total mem: {self.total_mem}MB")

    def plot(self):
        train_acc_index = -1
        test_acc_index = -1
        for i in range(len(self.header)):
            if self.header[i] == 'train_acc':
                train_acc_index = i
            elif self.header[i] == 'test_acc':
                test_acc_index = i

        assert (train_acc_index >= 0)
        data = np.array(self.table)
        x = np.arange(0, self.epochs)
        y = data[:, train_acc_index] if test_acc_index == -1 else [data[:, train_acc_index], data[:, test_acc_index]]
        plot_utils.plot(x, y, 'epoch', 'acc', ['train_acc', 'test_acc'])
        return


def format_table(table, header=None, name=None, separators=True, prefer_vertical_format=False, floatfmt="3.4f"):
    if header is not None:
        fmt_tb = tb.tabulate(table, header, floatfmt=floatfmt)
    elif table.__class__.__name__ == 'list':
        fmt_tb = tb.tabulate(table, floatfmt=floatfmt)
    elif table.__class__.__name__ == 'dict_items':
        keys = list(map(lambda x: x[0], table))
        values = list(map(lambda x: x[1], table))
        fmt_tb = tb.tabulate([values], keys, floatfmt=floatfmt)
    elif isinstance(table, dict):
        if prefer_vertical_format:
            fmt_tb = tb.tabulate(table.items(), ['Param', 'Val'], floatfmt=floatfmt)
        else:
            fmt_tb = tb.tabulate([table.values()], table.keys(), floatfmt=floatfmt)
    else:
        raise Exception('utils.format_table unknown class type: ' + table.__class__.__name__)

    if separators:
        print('*' * 20)
    if name is not None:
        print(name)
    print(fmt_tb)
    if separators:
        print('*' * 20)
        print()


def train_format_table_header(headers):
    row_format = "{:>10}  " * (len(headers))
    print(row_format.format(*headers))
    divider = (("-" * 10) + '  ') * (len(headers))
    print(divider)


def train_format_table_iteration(data):
    row_format = "{:>10}  " + "{:>10.2f}  " * (len(data) - 1)
    print(row_format.format(*data))
    print()


def format_row_carriage_return(header, row):
    assert (len(header) == len(row))
    print('', end='\r')
    for i in range(len(header)):
        if isinstance(row[i], float):
            print(f"{header[i]}: {row[i]:.2f},", end=' ')
        else:
            print(f"{header[i]}: {row[i]},", end=' ')
