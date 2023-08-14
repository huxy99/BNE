

from helper import *
import model.function as binary



class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.init_embed		= get_param((self.p.num_ent,   self.p.embed_dim))
		self.device		= self.edge_index.device
		self.init_rel = get_param((num_rel*2, self.p.embed_dim))

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))


	def forward_base(self, sub, rel):

		sub_emb	= torch.index_select(self.init_embed, 0, sub)
		rel_emb	= torch.index_select(self.init_rel, 0, rel)

		return sub_emb, rel_emb, self.init_embed


class BNE_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		
		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt

		self.rescale_H = flat_sz_h if self.p.bin_conv_rescale else 1 
		self.rescale_W = flat_sz_w if self.p.bin_conv_rescale else 1
		self.rescale_L = self.flat_sz if self.p.bin_lin_rescale else 1

		if self.p.bin_inputs=='bi':
			self.quantizer = binary.STEQuantizer()
		elif self.p.bin_inputs=='1-norm':
			self.quantizer = binary.NORMQuantizer()
		elif self.p.bin_inputs=='tanh':
			self.quantizer = binary.Tanh(lamda =self.p.lamda)
		elif self.p.bin_inputs=='leakyclip':
			self.quantizer = binary.Leakyclip()
		elif self.p.bin_inputs=='clip':
			self.quantizer = binary.Clip()
		else:
			self.quantizer = torch.nn.Identity()

		self.m_conv1		= binary.BinConv2d(1, out_channels=self.p.num_filt, kernel_size=self.p.ker_sz, stride=1, padding=0, bias=self.p.bias, binary_weights=self.p.bin_conv)
		self.lin		= binary.BinLinear(self.flat_sz, self.p.embed_dim, binary_weights=self.p.bin_lin)		
		
		if self.p.bin_rescale:
			self.rescale_conv = binary.LearnedRescaleLayer2d((1, self.p.num_filt, max(1, self.rescale_H), max(1, self.rescale_W)))
			self.rescale_lin = binary.LearnedRescaleLayer0d((1, self.p.embed_dim))
		else:
			self.rescale_conv = torch.nn.Identity()
			self.rescale_lin = torch.nn.Identity()
		
		if self.p.activ:
			self.activ = binary.PReLU()
		else:
			self.activ = torch.nn.ReLU()
		

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel)
		stk_inp				= self.concat(sub_emb, rel_emb) 
		x				= self.bn0(stk_inp)		
		x = self.hidden_drop(x)
		x = self.quantizer(x)
		x = self.m_conv1(x)
		x = self.rescale_conv(x) 
		x				= self.bn1(x)
		x				= self.activ(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x = self.quantizer(x)
		x				= self.lin(x) 
		x = self.rescale_lin(x)
		x				= self.bn2(x)
		x				= self.activ(x) 
		x				= self.hidden_drop2(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		return x


'''
from helper import *
from model.function import BinActive


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.init_embed		= get_param((self.p.num_ent,   self.p.embed_dim))
		self.device		= self.edge_index.device
		self.init_rel = get_param((num_rel*2, self.p.embed_dim))

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))


	def forward_base(self, sub, rel):

		x = self.init_embed
		r	= self.init_rel 
		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class BNE_CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
		self.weights = get_param((self.p.num_filt, 1, self.p.ker_sz, self.p.ker_sz))

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim) #修改

		


	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x = self.hidden_drop(x)
		x				= F.conv2d(x, self.weights,bias=None,stride=1,padding=0,dilation=1,groups=1)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score
'''
