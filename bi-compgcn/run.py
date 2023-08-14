from helper import *
from data_loader import *
import copy
from tqdm import tqdm
# sys.path.append('./')
from model.models import *

class Runner(object):

	def load_data(self):

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.data = ddict(list)
		sr2o = ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.edge_index, self.edge_type = self.construct_adj()

	def construct_adj(self):
		
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def __init__(self, params):
		
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
		self.logger.info("\nStudent Model:\n{}".format(str(self.model)))
		self.logger.info("\nStudent Optimizer:\n{}".format(str(self.optimizer)))


	def add_model(self, model, score_func):
		
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = BNE_CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = BNE_CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = BNE_CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)
		return model

	def add_teacher_model(self, model, score_func): #待修改
		
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = BNE_CompGCN_TransE(self.edge_index, self.edge_type, params=self.teacher_args)
		elif model_name.lower()	== 'compgcn_distmult': 	model = BNE_CompGCN_DistMult(self.edge_index, self.edge_type, params=self.teacher_args)
		elif model_name.lower()	== 'compgcn_conve': 	model = BNE_CompGCN_ConvE(self.edge_index, self.edge_type, params=self.teacher_args)
		else: raise NotImplementedError

		model.to(self.device)
		return model

	def read_batch(self, batch, split):
		
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def save_teacher_model(self, save_path):
		
		state = {
			'state_dict'	: self.teacher_model.state_dict(),
			'optimizer'	: self.teacher_optimizer.state_dict(),
			'args'		: vars(self.teacher_args)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])
	
	def load_teacher_model(self,load_path):
		state			= torch.load(load_path)
		state_dict		= state['state_dict']

		self.teacher_model.load_state_dict(state_dict)
		self.teacher_optimizer.load_state_dict(state['optimizer'])


	def evaluate(self, split, epoch):
		
		self.logger.info('Start {}ing Student Model Epoch{}'.format(split,epoch))
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		if split == 'test':
			self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\nMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\nhits@1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\nhits@3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\nhits@10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'], results['right_mr'], results['mr'], results['left_mrr'], results['right_mrr'], results['mrr'], results['left_hits@1'], results['right_hits@1'], results['hits@1'], results['left_hits@3'], results['right_hits@3'], results['hits@3'], results['left_hits@10'], results['right_hits@10'], results['hits@10']))
		else:
			self.logger.info('[Epoch {} {}]: MR: {:.5}; MRR: {:.5}; hits@1: {:.5}; hits@3: {:.5}; hits@10: {:.5}\n'.format(epoch, split,results['mr'], results['mrr'],results['hits@1'],results['hits@3'], results['hits@10']))
		return results


	def predict(self, split='valid', mode='tail_batch'):
		
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
			loop = tqdm(enumerate(train_iter), total =len(train_iter))
			for step, batch in loop:
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				

		return results


	def run_epoch(self, epoch, val_mrr = 0):

		self.logger.info('Start training Student Model Epoch{}'.format(epoch))
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])
		loop = tqdm(enumerate(train_iter), total =len(train_iter))
		for step, batch in loop:
			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')

			output	= self.model.forward(sub, rel)
			#loss	= self.model.loss(pred, label)

			if not self.p.no_distillation:
				teacher_output = self.teacher_model.forward(sub, rel)
				teacher_output = teacher_output.detach()
				#loss =	self.distillation(self.model, output, label, teacher_output, temp=self.p.temp, alpha=self.p.alpha)
				student_feature = torch.cat([self.model.state_dict()['init_embed'], self.model.state_dict()['init_rel']], dim=0)
				teacher_feature = torch.cat([self.teacher_model.state_dict()['init_embed'], self.teacher_model.state_dict()['init_rel']], dim=0)
				loss =	self.distillation(self.model, output, label, teacher_output, temp=self.p.temp, alpha=self.p.alpha,student_feature=student_feature,teacher_feature=teacher_feature,beta=self.p.beta)
			else:
				loss	= self.model.loss(torch.sigmoid(output), label)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}/{}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step,len(train_iter), np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def train_teacher(self, epoch):

		self.logger.info('Start training Teacher Model Epoch{}'.format(epoch))
		self.teacher_model.train()
		train_iter = iter(self.data_iter['train'])
		losses = []
		loop = tqdm(enumerate(train_iter), total =len(train_iter))
		for step, batch in loop:
			self.teacher_optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')
			output	= self.teacher_model.forward(sub, rel)
			pred = torch.sigmoid(output)
			loss	= self.teacher_model.loss(pred, label)

			loss.backward()
			self.teacher_optimizer.step()
			losses.append(loss.item())

		loss = np.mean(losses)
		self.logger.info('[Epoch {} training]:  Training Loss:{:.4}'.format(epoch, loss))

	def test_teacher(self, split, epoch):

		self.logger.info('Start testing Teacher Model Epoch{}'.format(epoch))
		left_results,left_loss  = self.predict_teacher(split=split, mode='tail_batch')
		right_results,right_loss = self.predict_teacher(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		loss = (left_loss+right_loss)/2
		self.logger.info('[Epoch {} {}]: MR: {:.5}; MRR: {:.5}; hits@1: {:.5}; hits@3: {:.5}; hits@10: {:.5}\n'.format(epoch, split,results['mr'], results['mrr'],results['hits@1'],results['hits@3'], results['hits@10']))

		return loss, results['mrr'], results['hits@10']
	
	def predict_teacher(self, split='test', mode=''):

		self.teacher_model.eval()

		results = {}
		losses = []
		test_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

		with torch.no_grad():
			loop = tqdm(enumerate(test_iter), total =len(test_iter))
			for step, batch in loop:
				sub, rel, obj, label	= self.read_batch(batch, split)
				output	= self.teacher_model.forward(sub, rel)
				pred = torch.sigmoid(output)
				loss	= self.teacher_model.loss(pred, label)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				
				losses.append(loss.item())

		loss = np.mean(losses)
		return results,loss

	def run_teacher(self):

		self.teacher_args = copy.deepcopy(self.p)
		self.teacher_args.bin_inputs = 'real'
		self.teacher_args.bin_lin = 'real'
		self.teacher_args.bin_rescale = False


		self.logger.info(vars(self.teacher_args))

		self.teacher_model = self.add_model(self.p.model, self.p.score_func)
		self.teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

		self.logger.info("\nTeacher Model:\n{}".format(str(self.teacher_model)))
		self.logger.info("\nTeacher Optimizer:\n{}".format(str(self.teacher_optimizer)))	

		teacher_history = []

		if self.p.teacher_restore:
			save_teacher_path = os.path.join('./pretrained', self.p.teacher_name)
			self.load_teacher_model(save_teacher_path)
			self.logger.info('Successfully Loaded previous teacher model\n')
			loss, mrr, hits10 = self.test_teacher('test', ' ')
			teacher_history.append((loss, mrr,hits10))
			
			return self.teacher_model, teacher_history
				
		self.teacher_args.name = self.p.teacher_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')				
		for epoch in range(self.teacher_args.epoch):
			teacher_train_loss = self.train_teacher(epoch)

			loss, mrr, hits10 = self.test_teacher('test', epoch)       
			teacher_history.append((loss, mrr,hits10))

		teacher_save_path = os.path.join('./pretrained', self.teacher_args.name)
		self.save_teacher_model(teacher_save_path)
		self.logger.info('Successfully Loaded teacher model\n')

		return self.teacher_model, teacher_history
	
	'''
	def distillation(self, model, y, labels, teacher_scores, temp, alpha):

		#return KLDivLoss()(torch.log(torch.sigmoid(y / temp)), torch.sigmoid(teacher_scores / temp)) * (
        #    temp * temp * alpha) + model.loss(torch.sigmoid(y), labels) * (1. - alpha)
		return KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp,dim=1)) * (
            temp * temp * alpha) + model.loss(torch.sigmoid(y), labels) * (1. - alpha)
	'''
	def distillation(self, model, y, labels, teacher_scores, temp, alpha,student_feature,teacher_feature,beta):
		loss1 =	model.loss(torch.sigmoid(y), labels) * (1. - alpha)	
		loss2 = KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp,dim=1)) * (temp * temp * alpha)
		loss3 = (student_feature - teacher_feature)**2 * ((student_feature > 0) | (teacher_feature > 0)).float()
		return loss1+loss2+torch.abs(loss3).sum()*beta
	

	def fit(self):
		
		if not self.p.no_distillation:#判断是否需要知识蒸馏
			self.teacher_model, self.teacher_history = self.run_teacher()

		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name)
		kill_cnt = 0
		for epoch in range(self.p.epoch):
			flag = 0
			train_loss  = self.run_epoch(epoch, val_mrr)
			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if flag == 0 and kill_cnt == 25: 
					flag = 1
					self.logger.info("Early Stopping!!")


		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='student_model',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='transe',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-epoch',		dest='epoch', 	type=int,       default=400,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=5,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-bin_inputs', dest='bin_inputs', default='bi', choices=['bi','1-norm','tanh','real','leakyclip','clip'], help='卷积/fc前的特征是否二值化')
	parser.add_argument('-lamda', dest='lamda', default=1, type=int, help='卷积/fc前的特征是否二值化')
	parser.add_argument('-bin_lin', dest='bin_lin', default='bi', choices=['bi','1-norm','real'], help='fc是否二值化')
	parser.add_argument('-bin_rescale',    	dest='bin_rescale', 		action='store_true',            help='线性层是否可伸缩') 	
	
	parser.add_argument('-no_distillation',    	dest='no_distillation', 		action='store_true', help='Learnable rescaling in the L dimension for the last conv before the MLP (# points)')
	parser.add_argument('-teacher_name',	default='teacher_model',			help='Set run name for saving/restoring models')
	parser.add_argument('-teacher_restore',         dest='teacher_restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-temp',  	dest='temp', 	default=7.0,  	type=float,	help='Dropout after GCN')
	parser.add_argument('-alpha',  	dest='alpha', 	default=0.3,  	type=float,	help='Dropout after GCN')
	parser.add_argument('-beta',  	dest='beta', default=1e-7, type=float,help='weight of feature loss')	

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	args = parser.parse_args()

	args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')


	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	model.fit()
