import copy

import log_utils
from log_utils import MetricLogger

import torch
import timm
from torch import nn
from torch.utils.data import DataLoader
import dataset_utils
import torchvision

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# todo how we can improve metriclogger and feedback during training:
#  https://albumentations.ai/docs/examples/pytorch_classification/ it has a really nice progress bar  multi-line
#  carriage return for log_utils: https://stackoverflow.com/questions/39455022/python-3-print-update-on-multiple
#  -lines this would allow us to give much better output during training.

# todo add arg channels_last see https://github.com/rwightman/pytorch-image-models/blob/master/train.py

# todo automatically create acc/loss charts for both test/train in wandb
#  https://docs.wandb.ai/guides/data-vis/log-tables


def get_dataloaders(train, test, batch_size=128, num_workers=0, cuda_stream_preload=True):
    train_dt = DataLoader(train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_dt = DataLoader(test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    if cuda_stream_preload:  # ~2% performance increase
        train_dt = dataset_utils.CudaPrefetchLoader(train_dt, device=device)
        test_dt = dataset_utils.CudaPrefetchLoader(test_dt, device=device)

    return train_dt, test_dt


def load_from_checkpoint(checkpoint_file, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return checkpoint['epoch']


def train_model_epoch(model, trainloader: DataLoader, acc_fn, criterion, optimizer, scaler, ema_model):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for data in trainloader:
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        # forward
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(inputs)
            loss = criterion(output, labels)

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()

        if ema_model is not None:
            ema_model.update(model)
        if acc_fn is not None:
            with torch.no_grad():
                running_acc += acc_fn(output, labels)

    running_acc /= len(trainloader)
    assert (running_acc <= 1)
    running_loss /= len(trainloader)
    return running_acc * 100, running_loss


def test_model(model, testloader: DataLoader, acc_fn=None, criterion=None, result_bag=False):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    if result_bag:
        results_bag = []
        labels_bag = []

    with torch.no_grad():
        for data in testloader:
            inputs =data[0].to(device=device)
            output = model(inputs)

            if criterion is not None:
                labels = data[1].to(device=device)
                loss = criterion(output, labels)
                running_loss += loss.item()
                if acc_fn is not None:
                    running_acc += acc_fn(output, labels)
            if result_bag:
                results_bag.append(output)
                labels_bag.append(data[1])

    if result_bag:
        results_bag = torch.cat(results_bag)
        labels_bag = torch.cat(labels_bag)

    running_acc /= len(testloader)
    assert (running_acc <= 1)
    running_loss /= len(testloader)
    if result_bag:
        return running_acc * 100, running_loss , labels_bag, results_bag
    else:
        return running_acc * 100, running_loss


def train_test_model(model, train_dt, test_dt=None, acc_fn=None, epochs=90,
                     loss=None, optimizer=None, lr_scheduler=None, ema_decay=0.0,
                     use_amp=True, check_point=None,
                     wandb=None, live_plot=True, live_log=True, print_report=False):
    # initialize metric logger which will report stats to terminal and draw a plot
    if test_dt is not None:
        user_header = ['train_acc', 'test_acc', 'train_loss', 'test_loss']
        live_plot_header = ['train_acc', 'test_acc']
    else:
        user_header = ['train_acc', 'train_loss']
        live_plot_header = ['train_acc']

    if wandb is not None and not isinstance(wandb, log_utils.WandbMetricLogger):
        wandb = log_utils.WandbMetricLogger(wandb)
        wandb.set_metrics(live_plot_header)
        wandb.training_start(model, train_dt, test_dt, epochs,
                             loss, optimizer, lr_scheduler, ema_decay)

    metric_logger = MetricLogger(
        user_header=user_header,
        live_plot=live_plot_header if live_plot else None,
        lr_scheduler=lr_scheduler,
        console_live_log=live_log,
        wandb=wandb
    )

    # initialize training variables
    model.to(device=device)
    if ema_decay:
        ema_model = timm.utils.ModelEmaV2(model, ema_decay, device)
    else:
        ema_model = None

    loss.to(device=device)
    best = None
    best_acc = -1
    best_epoch = -1
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    torch.cuda.empty_cache()  # maybe required to prevent memory leaks in notebook

    # run the training loop
    for epoch in metric_logger.log(range(epochs)):
        train_acc, train_loss = train_model_epoch(model, train_dt, acc_fn,
                                                  loss, optimizer, scaler,
                                                  ema_model)
        # decay learning rate
        if lr_scheduler is not None: lr_scheduler.step()

        if test_dt is not None:
            test_acc, test_loss = test_model(model, test_dt, acc_fn, loss)
            eval_acc = test_acc
            metric_logger.update(train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss)
        else:
            eval_acc = train_acc
            metric_logger.update(train_acc=train_acc, train_loss=train_loss)

        if eval_acc >= best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            best = model.state_dict()
            # torch.save(model, 'models/tmp.pt')

        if check_point is not None and epoch % 10:
            checkpoint = {
                'model': model.state_dict(),
                'ema_model': ema_model,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, check_point)

    if best is not None:
        model.load_state_dict(best)

    report = metric_logger.report(best_epoch, best_acc)
    if print_report:
        report.print()
    elif live_plot:
      report.brief()

    return report


def infer_model(model, test_dt):
    _, _, labels, result = test_model(model, test_dt, result_bag=True)
    labels = labels.cpu()
    result = result.cpu()
    return labels, result


def train_infer_model(model, train_dt, test_dt, acc_fn=None, epochs=90,
                      loss=None, optimizer=None, lr_scheduler=None,
                      use_amp=True, check_point=None, live_plot=True, print_report=True):
    train_report = train_test_model(model, train_dt, test_dt=None, acc_fn=acc_fn, epochs=epochs,
                                    loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler, use_amp=use_amp,
                                    check_point=check_point, live_plot=live_plot, print_report=print_report)

    result_bag = infer_model(model, test_dt)
    return train_report, result_bag


## Define accuracy functions
def classification_accuracy_fn(out, labels):
    _, preds = torch.max(out, dim=1)
    acc = torch.sum(preds == labels).item() / len(preds)
    assert (acc <= 1)
    return acc


def create_std_model(name='resnet50', output_features=None, scale_output_features=None,
                     pre_trained=True, feature_extraction=False):
    def __change_fc_attributes(model, last_name):
        if feature_extraction:
            set_parameter_requires_grad(model, False)

        # get the name of the last layer
        model_fc = getattr(model, last_name)
        if scale_output_features:

            if isinstance(model_fc, nn.Sequential):  # if the last layers are a group of Fully Connected
                for i in range(len(model_fc)):
                    class_i = model_fc[i]
                    if isinstance(class_i, nn.Linear):  # change only the FC layers
                        model_fc[i] = nn.Linear(int(class_i.in_features * scale_output_features),
                                                int(class_i.out_features * scale_output_features))
            else:  # it's just one FC
                model_fc = nn.Linear(int(model_fc.in_features * scale_output_features),
                                     int(model_fc.out_features * scale_output_features))
                setattr(model, last_name, model_fc)

        if output_features:
            if isinstance(model_fc, nn.Sequential):  # if the last layers are a group of Fully Connected
                last = len(model_fc) - 1
                model_fc[last] = nn.Linear(model_fc[last].in_features, output_features)
            else:  # it's just one FC
                model_fc = nn.Linear(model_fc.in_features, output_features)
                setattr(model, last_name, model_fc)

    if name == 'alexnet':
        model = torchvision.models.alexnet(pre_trained)
        __change_fc_attributes(model, 'classifier')

    elif name == 'googlenet':
        model = torchvision.models.googlenet(pre_trained)
        __change_fc_attributes(model, 'fc')

    elif name == 'inception':
        model = torchvision.models.inception_v3(pre_trained)
        if feature_extraction:
            set_parameter_requires_grad(model, False)
        if scale_output_features:
            raise NotImplementedError
        if output_features:
            model.AuxLogits.fc = nn.Linear(768, output_features)
            model.fc = nn.Linear(model.fc.in_features, output_features)
        # this seems to be trained with auxilary outputs :
        #  https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html see more
        # it requires input size to be 299,299

    elif name.startswith('resnet'):
        method_to_call = getattr(torchvision.models, name)
        model = method_to_call(pre_trained)
        __change_fc_attributes(model, 'fc')

    elif name == 'resnext50':
        model = torchvision.models.resnext50_32x4d(pre_trained)
        __change_fc_attributes(model, 'fc')

    elif name == 'resnext100':
        model = torchvision.models.resnext101_32x8d(pre_trained)
        __change_fc_attributes(model, 'fc')

    elif name.startswith('vgg') or name.startswith('densenet'):  # densenet161
        method_to_call = getattr(torchvision.models, name)
        model = method_to_call(pre_trained)
        __change_fc_attributes(model, 'classifier')

    elif name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(pre_trained)
        __change_fc_attributes(model, 'classifier')

    elif name == 'mobilenetv3s':
        model = torchvision.models.mobilenet_v3_small(pre_trained)
        __change_fc_attributes(model, 'classifier')

    elif name == 'mobilenetv3l':
        model = torchvision.models.mobilenet_v3_large(pre_trained)
        __change_fc_attributes(model, 'classifier')

    elif name == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pre_trained)
        if feature_extraction:
            set_parameter_requires_grad(model, False)
        if scale_output_features:
            model.classifier[1] = nn.Conv2d(int(model.classifier[1].in_channels / scale_output_features),
                                            int(model.classifier[1].out_channels / scale_output_features),
                                            kernel_size=1)
        if output_features:
            model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, output_features, kernel_size=1)
        raise NotImplementedError('Bad Implementation. Conv2D output_channels will not give N output_features')

    elif name == 'shufflenetv2':
        model = torchvision.models.shufflenet_v2_x1_0(pre_trained)
        __change_fc_attributes(model, 'fc')

    else:
        raise Exception("Unknown model name: " + name)

    model.to(device)

    return model


def set_parameter_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value
