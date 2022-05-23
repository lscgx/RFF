import torch.nn as nn
import numpy as np
import torch
import math
import argparse
import torch.optim as optim
import time
import os
import copy
import pickle
import random
import bisect
from scipy.stats import norm
from torchvision import datasets, models, transforms
from sympy import *

from utils import *
from models import *
from mask import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 & ImageNet Pruning')

parser.add_argument(
	'--data_dir',    
	default='G:\\data',    
	type=str,   
	metavar='DIR',                 
	help='path to dataset')
parser.add_argument(
	'--dataset',     
	default='CIFAR10',   
	type=str,   
	choices=('CIFAR10','ImageNet'),
	help='dataset')
parser.add_argument(
	'--num_workers', 
	default=0,           
	type=int,   
	metavar='N',                   
	help='number of data loading workers (default: 0)')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
	'--lr',         
	default=0.01,        
	type=float,                                
	help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    metavar='LR',
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    metavar='PATH',
    help='load the model from the specified checkpoint')
parser.add_argument(
	'--batch_size', 
	default=128, 
	type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
	'--momentum', 
	default=0.9, 
	type=float, 
	metavar='M',
    help='momentum')
parser.add_argument(
	'--weight_decay', 
	default=0., 
	type=float,
    metavar='W', 
    help='weight decay',
    dest='weight_decay')
parser.add_argument(
	'--gpu', 
	default='0', 
	type=str,
    help='GPU id to use.')
parser.add_argument(
    '--job_dir',
    type=str,
    default='',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','densenet_40','googlenet','mobilenet_v2'),
    help='The architecture to prune')
parser.add_argument(
    '--input_size',
    type=int,
    default=32,
    help='The num of input size')
parser.add_argument(
    '--save_id',
    type=int,
    default=0,
    help='save_id')
parser.add_argument(
    '--from_scratch',
    type=bool,
    default=False,
    help='train from_scratch')


args           = None
lr_decay_step  = None
logger         = None
compress_rate  = None
trainloader    = None
testloader     = None
criterion      = None
device         = None
model          = None
mask           = None
best_acc       = 0.
best_accs      = []

def init():
	global args,lr_decay_step,logger,compress_rate,trainloader,testloader,criterion,device,model,mask,best_acc,best_accs
	args = parser.parse_args()
	if args.lr_decay_step != 'cos':
		lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
	else :
		lr_decay_step = args.lr_decay_step
	logger = get_logger(os.path.join(args.job_dir, 'log/log'))
	compress_rate = format_compress_rate(args.compress_rate)
	trainloader,testloader = load_data(data_name = args.dataset, data_dir = args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model  = eval(args.arch)()
	mask   = eval('mask_'+args.arch)(model=model, job_dir=args.job_dir, device=device)
	best_acc = 0. 

	if len(args.job_dir) > 0  and args.job_dir[-1] != '\\':
		args.job_dir += '/'

	if len(args.gpu) > 1:
		gpus = args.gpu.split(',')
		device_id = []
		for i in gpus:
			device_id.append(int(i))
		print('device_ids:',device_id)
		model = nn.DataParallel(model, device_ids=device_id).cuda()
	else :
		model = model.to(device)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.info('args:{}'.format(args))

def train(epoch,model,cov_id,trainloader,optimizer,criterion,mask = None):
	losses = AverageMeter('Loss', ':.4f')
	top1   = AverageMeter('Acc@1', ':.2f')
	top5   = AverageMeter('Acc@5', ':.2f')

	model.train()
	num    = len(trainloader)
	since  = time.time()
	_since = time.time()
	for i, (inputs,labels) in enumerate(trainloader, 0):

		# if i > 1 : break

		inputs = inputs.to(device)
		labels = labels.to(device)

		if lr_decay_step == 'cos': adjust_learning_rate(optimizer, epoch, i, num)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		if mask is not None : mask.grad_mask(cov_id)

		acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(acc1[0], inputs.size(0))
		top5.update(acc5[0], inputs.size(0))

		if i!=0 and i%2000 == 0:   #2000
			_end = time.time()
			logger.info('epoch[{}]({}/{}) Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(epoch,i,int(1280000/args.batch_size),losses.avg,top1.avg,top5.avg,_end - _since))
			_since = time.time()

	end = time.time()
	logger.info('train    Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

def validate(epoch,model,cov_id,testloader,criterion,save = True):
	losses = AverageMeter('Loss', ':.4f')
	top1 = AverageMeter('Acc@1', ':.2f')
	top5 = AverageMeter('Acc@5', ':.2f')

	model.eval()
	with torch.no_grad():
		since = time.time()
		for i, data in enumerate(testloader, 0):
			# if i > 1 : break

			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1[0], inputs.size(0))
			top5.update(acc5[0], inputs.size(0))

		end = time.time()
		logger.info('validate Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

	global best_acc
	if save and best_acc <= top1.avg:
		best_acc = top1.avg
		state = {
			'state_dict': model.state_dict(),
			'best_prec1': best_acc,
			'epoch': epoch
		}
		if not os.path.isdir(args.job_dir + 'pruned_checkpoint'):
			os.makedirs(args.job_dir + 'pruned_checkpoint')
		cov_name = '_cov' + str(cov_id)
		if cov_id == -1: cov_name = ''
		torch.save(state,args.job_dir + 'pruned_checkpoint/'+args.arch+cov_name + '.pt')
		logger.info('storing checkpoint:'+'pruned_checkpoint/'+args.arch+cov_name + '.pt')

	return top1.avg,top5.avg

def iter_vgg16bn():
	cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
	last_layer = 12
	ranks      = []
	optimizer  = None 
	scheduler  = None
	nxt_rank   = None
	conv_names = get_conv_names(model)

	for layer_id in range(last_layer,-1,-1):
		logger.info("===> pruning layer {}".format(layer_id))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

		if layer_id == last_layer: 
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
		else :
			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id+1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id+1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		_num = len(model.state_dict()[conv_names[layer_id]])
		if layer_id < last_layer: _nxt_num = len(model.state_dict()[conv_names[layer_id+1]])
		relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=layer_id)

		effects_pct = [0.]*_num
		if layer_id == last_layer: 
			effects_pct = relu_expect
		else :
			for next_feature_id in nxt_rank:
				_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect)
				effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
		rank = np.argsort(effects_pct)
		rank1 = rank[int(_num*compress_rate[layer_id]):_num] 
		ranks.insert(0,rank1)
		nxt_rank = rank1
		mask.layer_mask(layer_id, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
		ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		_train(layer_id)

	logger.info(best_accs)

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(0) + '.pt', map_location=device)
	rst_model = vgg_16_bn(_state_dict(final_state_dict['state_dict']),ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)
	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_resnet_56():
	ranks = []
	layers = 55

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	#------------------------------------------------
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	ranks = ['no_pruned']*layers
	for layer_id in range(1,4):
		rank1 = []
		_id = (layer_id-1)*18 + 0 * 2 + 2 
		_num = len(model.state_dict()[conv_names[_id]])
		if layer_id == 0 and compress_rate[0] > 0.: 
			prune_num = int(_num*compress_rate[0])
			rank1 = list(range(_num))[prune_num//2:_num-(prune_num-prune_num//2)]
			mask.layer_mask(0, param_per_cov=3, rank=rank1, type = 1, arch=args.arch) # the first layer
			ranks[0] = rank1
		for block_id in range(0,9):
			_id = (layer_id-1)*18 + block_id * 2 + 2 
			prune_num = int(_num*compress_rate[_id])
			rank1 = list(range(_num))[prune_num//2:_num-(prune_num-prune_num//2)]
			if compress_rate[_id] > 0.:
				mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
			ranks[_id] = rank1
		_train(_id)
	#------------------------------------------------
	_id = layers - 1
	for layer_id in range(3,0,-1):

		for block_id in range(8,-1,-1):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

			for cov_id in range(2,0,-1):
				if cov_id == 1:
					_id = (layer_id-1)*18 + block_id * 2 + cov_id 
					_num = len(model.state_dict()[conv_names[_id]])
					relu_expect = get_relu_expect(arch='resnet_56', model=model, layer_id=_id)

					effects_pct = [0.]*_num
					for next_feature_id in ranks[_id+1]: 
						_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,_id,next_feature_id,relu_expect)
						effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
					rank = np.argsort(effects_pct)
					rank1 = rank[int(_num*compress_rate[_id]):_num] 
					ranks[_id] = rank1
					mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
			ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			_train(_id)

	logger.info(best_accs)
	logger.info(compress_rate)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(1) + '.pt', map_location=device)
	rst_model = resnet_56(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks=ranks)
	logger.info(rst_model)

	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_mobilenet_v2():
	ranks = []

	cfg = model.interverted_residual_setting
	n_blocknum = [0,1,2,3,4,3,3,1]
	_compress_rate = [0]
	IR_output = [32]
	IR_expand = [1]
	sequence_id = [0] #sequence_last_block
	for i in range(1,7+1):
		_compress_rate += [compress_rate[i]] * n_blocknum[i]
		IR_output += [cfg[i-1][1]] * n_blocknum[i]
		IR_expand += [cfg[i-1][0]] * n_blocknum[i]
		sequence_id += [i] * n_blocknum[i]

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer  = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	ranks.insert(0,'no_pruned')#51 IR-18
	_id = 50
	is_sequence_last = -1 # 
	sequence_last_rank = None 
	sequence_2th_rank = None 
	for IR_id in range(17,0,-1):
		logger.info("===> pruning InvertedResidual block {}".format(IR_id))

		if IR_id == 17:
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(pruned_checkpoint)
		else :
			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(IR_id+1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(IR_id+1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		# acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		for cov_id in range(3,0,-1):
			if IR_id == 1 and cov_id < 3: break

			if cov_id == 3:
				relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=_id,relu=False)
			elif cov_id == 2:
				relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=_id)

			if cov_id > 1:

				_num = len(model.state_dict()[conv_names[_id]])
				_nxt_num = len(model.state_dict()[conv_names[_id+1]])

				effects_pct = [0.]*_num

				for next_feature_id in range(_nxt_num):
					_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,_id,next_feature_id,relu_expect)
					effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
				rank = np.argsort(effects_pct)
				
				if cov_id == 2:
					c_output_num = math.ceil(IR_output[IR_id-1] * (1. - _compress_rate[IR_id-1])) * IR_expand[IR_id]  #IR_expand
					rank1 = rank[_num-c_output_num:_num] 
					sequence_2th_rank = rank1
				else :
					if sequence_id[IR_id] != is_sequence_last:
						rank1 = rank[int(_num*_compress_rate[IR_id]):_num] 
						is_sequence_last = sequence_id[IR_id]
						sequence_last_rank = rank1
					else :
						rank1 = sequence_last_rank
			else :
				rank1 = sequence_2th_rank

			ranks.insert(0,rank1)
			mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)

			_id -= 1

		ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
		_train(IR_id)

	ranks.insert(0,'no_pruned') # 1
	ranks.insert(0,'no_pruned') # 0

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov1.pt', map_location=device)
	rst_model = mobilenet_v2(compress_rate=compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_resnet_50():
	ranks = []
	layers = 49

	stage_repeat = [3, 4, 6, 3]
	layer_last_id = [0,10,23,42,52]
	branch_types  = [-1,2,1,0,0]

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	_id = 52
	for layer_id in range(4,0,-1):
		if layer_id == 4:
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(pruned_checkpoint)
		else :
			pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])
		lid    = layer_last_id[layer_id]
		_num   = len(model.state_dict()[conv_names[lid]])
		rank   = get_rank_resnet_50(model,lid,branch_type=branch_types[layer_id])
		l_rank = rank[int(_num*compress_rate[lid-layer_id]):_num] 
		cid    = 0
		for block_id in range(0,stage_repeat[layer_id-1]):
			if block_id == 0:
				mask.layer_mask(layer_last_id[layer_id-1]+3, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				mask.layer_mask(layer_last_id[layer_id-1]+4, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				cid = layer_last_id[layer_id-1]+4
			else :
				cid += 3
				mask.layer_mask(cid, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
		acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
		_train(lid)

		for block_id in range(stage_repeat[layer_id-1]-1,-1,-1):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			if block_id == stage_repeat[layer_id-1]-1:
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])
			else :
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])
			block_conv2_rank = None
			for cov_id in range(3,0,-1):
				if block_id == 0 and cov_id == 3: 
					_id -= 1
					ranks.insert(0,l_rank)

				_num = len(model.state_dict()[conv_names[_id]])

				cpid = _id - layer_id # 0 - 48
				if block_id == 0 : cpid += 1
				if cov_id < 3:
					if cov_id == 2:
						rank = get_rank_resnet_50(model,_id,nxt_rank=l_rank)
					else :
						rank = get_rank_resnet_50(model,_id,nxt_rank=block_conv2_rank)
					rank1 = rank[int(_num*compress_rate[cpid]):_num] 
					ranks.insert(0,rank1)
					if cov_id == 2: block_conv2_rank = rank1
					mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
					acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
				else :
					ranks.insert(0,l_rank)
				_id -= 1
			_train(_id)
			logger.info("===> layer_id {} block_id {} bestacc {:.4f}".format(layer_id,block_id,best_accs[-1]))
	ranks.insert(0,'no_pruned')

	logger.info(best_accs)

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov0.pt', map_location=device)
	rst_model = resnet_50(compress_rate=compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_googlenet():
	ranks = []
	inceptions = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']
	branch_offset = [0,1,3,6] #[1,2,4,7]

	layer_0_id = [1]
	for i in range(8):
		layer_0_id.append(layer_0_id[-1]+7)
	ffmaps = []
	for x in model.filters:
		ffmaps.append([sum(x[:0]),sum(x[:1]),sum(x[:2]),sum(x[:3]),sum(x[:4])])

	conv_names = get_conv_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	state_dict = model.state_dict()
	relu_expect = get_relu_expect(arch = args.arch, model = model, layer_id = 0)

	#--------------------------get first-layer rank-----------------------------
	_num = len(state_dict[conv_names[0]])
	effects_pct = [0.]*_num
	for nxt_oft in branch_offset:
		_nxt_num = len(state_dict[conv_names[nxt_oft+layer_0_id[0]]])
		for next_feature_id in range(_nxt_num):
			_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,0,next_feature_id,relu_expect,next_layer_id=nxt_oft+layer_0_id[0])
			effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
	rank = np.argsort(effects_pct)
	rank1 = rank[int(_num*compress_rate[0]):_num] 
	ranks.append([rank1])
	#---------------------------------------------------------------
	logger.info("===> pruning pre_layers")
	mask.layer_mask(0, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
	_train(0)

	for i,inception_id in enumerate(inceptions):
		# i += 1
		logger.info("===> pruning inception_id {}".format(i))

		pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(i) + '.pt', map_location=device)
		logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(i) + '.pt')
		model.load_state_dict(pruned_checkpoint['state_dict'])

		i_offset = [2,4,5]
		_rank = []

		for j in range(3):
			layer_id = (i)*7 + 1 + i_offset[j]
			_num = len(model.state_dict()[conv_names[layer_id]])
			if compress_rate[(i)*3+j+1] == 0.:
				_rank.append(list(range(_num)))	 
				continue
			relu_expect = get_relu_expect(arch = args.arch, model = model, layer_id = layer_id)

			effects_pct = [0.]*_num
			if j == 1:
				_nxt_num = len(state_dict[conv_names[layer_id+1]])
				for next_feature_id in range(_nxt_num):
					_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=layer_id+1)
					effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
					pass
			else :
				for k,nxt_oft in enumerate(branch_offset):
					_nxt_num = len(state_dict[conv_names[nxt_oft+layer_0_id[i+1]]])
					ffmaps_offset_id = 1 if j == 0 else 2
					for next_feature_id in range(_nxt_num):
						_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=nxt_oft+layer_0_id[i+1],offset=ffmaps[i][ffmaps_offset_id])
						effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
						pass

			rank = list(np.argsort(effects_pct))
			rank1 = rank[int(_num*compress_rate[(i)*3+j+1]):_num]
			_rank.append(rank1)
			mask.layer_mask(layer_id, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)

		ranks.append(_rank)
		_train(i+1)
		logger.info("===> inception_id {} best_acc {:.4f}".format(i,best_accs[-1]))

	logger.info(best_accs)

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov9.pt', map_location=device)
	rst_model =  googlenet(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_densenet40():
	ranks = []

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer  = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	growthRate = 12
	layer_id   = 39-1
	total_features = 24 + 12 * 36

	for dense_id in range(3,0,-1):
		for block_id in range(11,-1,-1):
			logger.info("===> pruning dense_id {} block_id {}".format(dense_id,block_id))

			if dense_id == 3 and block_id == 11:
				pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
				logger.info('loading checkpoint:' + args.resume)
				model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
			else :
				pruned_checkpoint = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id+1) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id+1) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
			state_dict = model.state_dict()

			_num = len(state_dict[bn_names[layer_id]+'.weight'])

			effects_pct = [0.]*growthRate
			relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=layer_id)

			if layer_id == 38: 
				effects_pct = relu_expect[total_features-growthRate:]
			else :
				for next_feature_id in nxt_rank:   #offset=_num-growthRate,
					_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,_start=_num-growthRate)
					effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
			
			rank = list(np.argsort(effects_pct))
			rank1 = rank[int(growthRate*compress_rate[layer_id]):growthRate]
			ranks.insert(0,rank1)
			nxt_rank = rank1
			mask.layer_mask(layer_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)

			_train(layer_id)

			layer_id -= 1

		# trans
		if dense_id > 1:
			logger.info("===> pruning trans_id {}".format(dense_id-1))
			_num = len(state_dict[bn_names[layer_id]+'.weight'])
			effects_pct = [0.]*_num
			relu_expect = get_relu_expect(arch=args.arch,model=model,layer_id=layer_id)
			for next_feature_id in nxt_rank:
				_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect)
				effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
			rank = list(np.argsort(effects_pct))
			rank1 = rank[int(_num*compress_rate[layer_id]):_num]
			ranks.insert(0,rank1)
			nxt_rank = rank1
			mask.layer_mask(layer_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
			_train(layer_id)
			layer_id -= 1

	ranks.insert(0,'no_pruned')

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'_cov1.pt', map_location=device)
	rst_model = densenet_40(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)
	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def train_from_scratch():
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	if args.compress_rate is None:
		compress_rate = pruned_checkpoint['compress_rate']
	else :
		compress_rate = format_compress_rate(args.compress_rate)
	model = eval(args.arch)(compress_rate=compress_rate).to(device)
	logger.info(model)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	validate(0,model,0,testloader,criterion,save = False)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	if lr_decay_step != 'cos':
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	else :
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0.)

	for epoch in range(0, args.epochs):
		if lr_decay_step !='cos' and epoch in lr_decay_step[-1:] and args.arch not in ['resnet_50','mobilenet_v2']: #!= 'resnet_50' 'resnet_56',
			resume = args.job_dir + 'pruned_checkpoint/'+args.arch+'.pt'
			pruned_checkpoint = torch.load(resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + 'pruned_checkpoint/'+args.arch+'.pt')
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion) #,mask
		if lr_decay_step != 'cos': scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	flops,params = model_size(model,args.input_size,device)

	best_model = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'.pt', map_location=device)
	rst_model = eval(args.arch)(compress_rate=compress_rate).to(device)
	rst_model.load_state_dict(_state_dict(best_model['state_dict']))

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	torch.save(state,args.job_dir + 'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def _train(i):
	global best_acc,best_accs
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch, model,i,trainloader,optimizer,criterion,mask) #,mask
		scheduler.step()
		validate(epoch,model,i,testloader,criterion)
	if args.epochs > 0 and best_acc > 0.:
		best_accs.append(round(best_acc.item(),4))
	else:
		best_accs.append(0.)
	best_acc=0.

def get_conv_names(model = None):
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.Conv2d):
			conv_names.append(name+'.weight')
	return conv_names

def get_bn_names(model = None):
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.BatchNorm2d):
			conv_names.append(name)
	return conv_names

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

def adjust_learning_rate(optimizer, epoch, step, len_iter):

	if lr_decay_step == 'cos':  
		lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

	if epoch < 5:
		lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def get_relu_expect(arch = 'vgg_16_bn', model = None, layer_id = 0,relu = True):
	relu_expect  = []
	state_dict = model.state_dict()
	if arch == 'vgg_16_bn':
		cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
		norm_weight = state_dict['features.norm'+str(cfg[layer_id])+'.weight']
		norm_bias = state_dict['features.norm'+str(cfg[layer_id])+'.bias']
	elif arch in ['resnet_56','googlenet','densenet_40','mobilenet_v2']:
		bn_names = get_bn_names(model)
		name = bn_names[layer_id]
		norm_weight = state_dict[name+'.weight']
		norm_bias  = state_dict[name+'.bias']

	num = len(norm_bias)
	for i in range(num):
		if norm_weight[i].item() < 0.: 
			norm_weight[i] = 1e-5
		if relu == True:
			_norm = norm(norm_bias[i].item(),norm_weight[i].item())
			relu_expect.append(_norm.expect(lb=0))
		else :
			relu_expect.append(norm_bias[i].item())
	return relu_expect

def get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=None,offset=0,_start=0):
	conv_names = get_conv_names(model=model)
	bn_names = get_bn_names(model=model)
	num = len(model.state_dict()[bn_names[layer_id]+'.weight'])
	if next_layer_id is None: next_layer_id = layer_id + 1
	next_state_dict_conv = model.state_dict()[conv_names[next_layer_id]]
	effects = []
	eps = 1e-7
	_sum = 0.
	_abs_sum = 0.
	for i in range(_start,num):
		effect = next_state_dict_conv[next_feature_id][i+offset].sum().item() * relu_expect[i]
		_sum  += effect
		_abs_sum += abs(effect)
		effects.append(effect)
	effects_pct = [abs(x/(_abs_sum+eps)) for x in effects]
	return effects,_sum,_abs_sum,effects_pct

def get_relu_expect_resnet_50(model,layer_id):
	relu_expect = []
	block_0_last_id = [3,13,26,45]
	block_dsamp_last_id = [4,14,27,46]
	block_last_id = [7]+[17,20]+[30,33,36,39]+[49]+[10,23,42]

	state_dict = model.state_dict()
	bn_names   = get_bn_names(model)

	if layer_id in block_0_last_id:
		norm_weight_1 = state_dict[bn_names[layer_id]+'.weight']
		norm_bias_1   = state_dict[bn_names[layer_id]+'.bias']
		norm_weight_2 = state_dict[bn_names[layer_id+1]+'.weight']
		norm_bias_2   = state_dict[bn_names[layer_id+1]+'.bias']
		num = len(norm_bias_1)
		for i in range(num):
			if norm_weight_1[i].item() < 0.: norm_weight_1[i] = 1e-5
			if norm_weight_2[i].item() < 0.: norm_weight_2[i] = 1e-5	
			norm_1 = norm(norm_bias_1[i].item(),norm_weight_1[i].item())
			norm_2 = norm(norm_bias_2[i].item(),norm_weight_2[i].item())
			mu     = norm_bias_1[i].item()+norm_bias_2[i].item()
			sigma  = sqrt((norm_weight_1[i].item())**2+(norm_weight_2[i].item())**2)
			norm_3 = norm(mu,float(sigma))
			relu_expect.append(norm_3.expect(lb=0)) #
	elif layer_id in block_dsamp_last_id:
		return get_relu_expect_resnet_50(model,layer_id-1)
	elif layer_id in block_last_id:
		relu_expect_1 = get_relu_expect_resnet_50(model,layer_id-3)
		norm_weight   = state_dict[bn_names[layer_id]+'.weight']
		norm_bias     = state_dict[bn_names[layer_id]+'.bias']
		num = len(norm_bias)
		for i in range(num):
			if norm_weight[i].item() < 0.: norm_weight[i] = 1e-5
			_norm = norm(norm_bias[i].item(),norm_weight[i].item())
			relu_expect.append(_norm.expect(lb=0)+relu_expect_1[i])
	else :
		norm_weight = state_dict[bn_names[layer_id]+'.weight']
		norm_bias   = state_dict[bn_names[layer_id]+'.bias']
		num = len(norm_bias)
		for i in range(num):
			if norm_weight[i].item() < 0.: norm_weight[i] = 1e-5
			_norm = norm(norm_bias[i].item(),norm_weight[i].item())
			relu_expect.append(_norm.expect(lb=0))

	return relu_expect

def get_rank_resnet_50(model,layer_id,branch_type=0,nxt_rank=None,skip_branch_rank=None): # branch_type {0,1,2}
	layer_last_id = [0,10,23,42,52]
	block_0_last_id = [3,13,26,45]
	block_last_id = [7]+[17,20]+[30,33,36,39]+[49]
	block_dsamp_last_id = [4,14,27,46]

	conv_names  = get_conv_names(model)
	_num        = len(model.state_dict()[conv_names[layer_id]])
	state_dict  = model.state_dict()
	effects_pct = [0.]*_num
	last_layer  = 52

	if layer_id != last_layer and nxt_rank is None:
		nxt_rank = list(range(len(state_dict[conv_names[layer_id+1]])))

	if layer_id == last_layer:
		relu_expect = get_relu_expect_resnet_50(model,layer_id)
		effects_pct = relu_expect
	elif layer_id in layer_last_id:
		relu_expect = get_relu_expect_resnet_50(model,layer_id)
		if branch_type in [0,2]:
			for next_feature_id in nxt_rank:
				_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=layer_id+1)
				effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
		if skip_branch_rank is None:
			skip_branch_rank = list(range(len(state_dict[conv_names[layer_id+4]])))	
		if branch_type in [1,2]:
			for next_feature_id in skip_branch_rank:
				_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=layer_id+4)
				effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]
	elif layer_id in block_0_last_id+block_last_id+block_dsamp_last_id:
		_layer_id = layer_last_id[bisect.bisect_right(layer_last_id,layer_id)]
		return get_rank_resnet_50(model,_layer_id,branch_type=branch_type)
	else :
		relu_expect = get_relu_expect_resnet_50(model,layer_id)
		for next_feature_id in nxt_rank:
			_,_,_,_effects_pct = get_effect_for_dstr_single_preL(model,layer_id,next_feature_id,relu_expect,next_layer_id=layer_id+1)
			effects_pct = [x+y for x,y in zip(effects_pct,_effects_pct)]

	rank = list(np.argsort(effects_pct))
	return rank

if __name__ == '__main__':

	init()

	if args.from_scratch is True:
		train_from_scratch()
	else :
		if args.arch == 'vgg_16_bn':
			iter_vgg16bn()
		elif args.arch == 'resnet_56':
			iter_resnet_56()
		elif args.arch == 'resnet_50':
			iter_resnet_50()
		elif args.arch == 'densenet_40':
			iter_densenet40()
		elif args.arch == 'googlenet':
			iter_googlenet()
		elif args.arch == 'mobilenet_v2':
			iter_mobilenet_v2()




