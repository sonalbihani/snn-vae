import time
import logging
from data_builder import *
from networks_for_CIFAR import *
from networks_for_ImageNet import *
from utils import accuracy, AvgrageMeter, save_checkpoint, get_model, create_para_dict, read_param, record_param, deletStrmodule, randomize_gate
import sys
sys.path.append("..")
from layers import *
from tensorboardX import SummaryWriter
from torch.cuda import amp
from schedulers import *
from Regularization import *
import random
from tqdm import tqdm

####################################################
# args                                             #
#                                                  #
####################################################
class Args:
    def __init__(self, eval=False, eval_resume='./raw/models', train_resume='./raw/models', batch_size=16, epochs=50, 
                 learning_rate=1e-1, momentum=0.9, weight_decay=4e-5, seed=9, auto_continue=False, display_interval=10, 
                 save_interval=10, dataset_path='./dataset/', train_dir='./fminst/train', val_dir='./fmnist/val', 
                 tunable_lif=False, amp=False, modeltag='SNN', gate=[0.6, 0.8, 0.6], static_gate=False, static_param=False, 
                 channel_wise=False, softsimple=False, soft_mode=False, t=3, randomgate=False, fmnist=True, celeba=False):
        self.eval = eval
        self.eval_resume = eval_resume
        self.train_resume = train_resume
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.seed = seed
        self.auto_continue = auto_continue
        self.display_interval = display_interval
        self.save_interval = save_interval
        self.dataset_path = dataset_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.tunable_lif = tunable_lif
        self.amp = amp
        self.modeltag = modeltag
        self.gate = gate
        self.static_gate = static_gate
        self.static_param = static_param
        self.channel_wise = channel_wise
        self.softsimple = softsimple
        self.soft_mode = soft_mode
        self.t = t
        self.randomgate = randomgate
        self.fminst = fmnist
        self.celeba = celeba


####################################################
# trainer & tester                                 #
#                                                  #
####################################################
log_interval = 200

def train(args, model, device, train_loader, optimizer, epoch, writer, criterion, scaler=None):
    layer_cnt, gate_score_list = None, None
    t1 = time.time()
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        if scaler is not None:
            with amp.autocast():
                output = model(data)
                loss = criterion(output, target)
                train_loss+=loss

        else:
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    average_train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, average_train_loss))
    return average_train_loss



def test(args, model, device, test_loader, epoch, writer, criterion, modeltag, dict_params, best= None):
    layer_cnt, gate_score_list = None, None
    test_loss = 0
    model.eval()# inactivate BN
    t1 = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            loss = criterion(output, target)
            test_loss+=loss

        record_param(args, model, dict=dict_params, epoch=epoch, modeltag=modeltag)
    average_test_loss = test_loss / len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(average_test_loss))
    return average_test_loss

def seed_all(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    args = Args()

    seed_all(args.seed)

    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    epochs = 1#已经迭代的次数
    initial_dict = {'gate': [0.6, 0.8, 0.6], 'param': [tau, Vth, linear_decay, conduct],
                   't': steps, 'static_gate': True, 'static_param': False, 'time_wise': True, 'soft_mode': False}
    initial_dict['gate'] = args.gate
    initial_dict['static_gate'] = args.static_gate
    initial_dict['static_param'] = args.static_param
    initial_dict['time_wise'] = False
    initial_dict['soft_mode'] = args.soft_mode
    if args.t != steps:
        initial_dict['t']=args.t

    # In case time step is too large, we intuitively recommend to use the following code to alleviate the linear decay
    # initial_dict['param'][2] = initial_dict['param'][1]/(initial_dict['t'] * 2)


    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    if args.fminst:
        train_loader, val_loader, _ = build_data(dataset='FashionMNIST',
                                                 batch_size=args.batch_size, train_val_split=False, workers=2,
                                                 fmnist_train_dir=args.train_dir, fmnist_val_dir=args.val_dir)
    else:
        train_loader, val_loader, _ = build_data(dataset='CelebA', dpath=args.dataset_path,
                                                 batch_size=args.batch_size, train_val_split=False, workers=2)

    print('load data successfully')

    print(initial_dict)
    #prepare the model
    if args.fmnist:
        model = ResNet_34_stand_CW(lif_param=initial_dict, input_size=224, n_class=1000)

    elif args.celeba:
        model = ResNet_34_stand_CW(lif_param=initial_dict, input_size=32, n_class=100)

    else: #cifar10
         model = ResNet_34_stand_CW(lif_param=initial_dict, input_size=32, n_class=100)


    if args.randomgate:
        randomize_gate(model)
        # model.randomize_gate
        print('randomized gate')

    modeltag = args.modeltag
    writer = SummaryWriter('./summaries/' + modeltag)
    print(model)
    dict_params = create_para_dict(args, model)
    # recording the initial GLIF parameters
    record_param(args, model, dict=dict_params, epoch=0, modeltag=modeltag)
    # classify GLIF-related params
    choice_param_name = ['alpha', 'beta', 'gamma']
    lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
    all_params = model.parameters()
    lif_params = []
    lif_choice_params = []
    lif_cal_params = []

    for pname, p in model.named_parameters():
        if pname.split('.')[-1] in choice_param_name:
            lif_params.append(p)
            lif_choice_params.append(p)
        elif pname.split('.')[-1] in lifcal_param_name:
            lif_params.append(p)
            lif_cal_params.append(p)
    # fetch id
    params_id = list(map(id, lif_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # optimizer & scheduler
    if args.tunable_lif:
        init_lr_diff = 10
        optimizer = torch.optim.SGD([
                {'params': other_params},
                {'params': lif_cal_params, "weight_decay": 0.},
                {'params': lif_choice_params, "weight_decay": 0., "lr":args.learning_rate / init_lr_diff}
            ],
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        scheduler = CosineAnnealingLR_Multi_Params_soft(optimizer,
                                                            T_max=[args.epochs, args.epochs, int(args.epochs)])
    else:
        optimizer = torch.optim.SGD([
            {'params': other_params},
            {'params': lif_params, "weight_decay": 0.}
        ],
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = Loss(args)
    device = torch.device("cuda" if use_gpu else "cpu")
    #Distributed computation
    if torch.cuda.is_available():
        loss_function = criterion.cuda()
    else:
        loss_function = criterion.cpu()

    if args.auto_continue:
        lastest_model = get_model(modeltag)
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location='cpu')
            epochs = checkpoint['epoch']
            if torch.cuda.device_count() > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                checkpoint = deletStrmodule(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint, the epoch is {}'.format(epochs))
            dict_params = read_param(epoch=epochs, modeltag=modeltag)
            for i in range(epochs):
                scheduler.step()
            epochs += 1


    best = {'acc': 0., 'epoch': 0}

    if args.eval:
        lastest_model = get_model(modeltag, addr=args.eval_resume)
        if lastest_model is not None:
            epochs = -1
            checkpoint = torch.load(lastest_model, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if torch.cuda.device_count() > 1:
                device = torch.device(local_rank)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=False)
            else:
                model = model.to(device)
            test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        else:
            print('no model detected')
        exit(0)


    if torch.cuda.device_count() > 1:
        device = torch.device(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=False)
    else:
        model = model.to(device)


    print('the random seed is {}'.format(args.seed))

    # amp
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    for t in tqdm(range(args.epochs)):
        train(args, model, device, train_loader, optimizer, epochs, writer, criterion=loss_function,
              scaler=scaler)
        if t % 1 == 0:
            test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        else:
            pass
        print(f"Epoch {t+1}:\t Train accuracy: {100*train_accuracy:0.1f}%\t Avg train loss: {average_train_loss:>6f}\t Test accuracy: {100*test_accuracy:0.1f}%\t Avg test loss: {average_test_loss:>6f}")
        print('and lr now is {}'.format(scheduler.get_last_lr()))
        scheduler.step()
    writer.close()



if __name__ == "__main__":
    main()
