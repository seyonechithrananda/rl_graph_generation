import tensorflow as tf
import numpy as np
import gym
import gym_molecule
from baselines.common.distributions import make_pdtype,MultiCatCategoricalPdType,CategoricalPdType
import baselines.common.tf_util as U


# gcn mean aggregation over edge features
def GCN(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple'):
    '''
    state s: (adj,node_feature)
    :param adj: b*n*n
    :param node_feature: 1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[0]
    in_channels = node_feature.get_shape()[-1]
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [edge_dim, in_channels, out_channels])
        b = tf.compat.v1.get_variable("b", [edge_dim, 1, out_channels])
        node_embedding = adj@tf.compat.v1.tile(node_feature,[edge_dim,1,1])@W+b
        if is_act:
            node_embedding = tf.compat.v1.nn.relu(node_embedding)
        # todo: try complex aggregation
        node_embedding = tf.compat.v1.reduce_mean(node_embedding,axis=0,keepdims=True) # mean pooling
        if is_normalize:
            node_embedding = tf.compat.v1.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

# gcn mean aggregation over edge features
def GCN_batch(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple',aggregate='sum'):
    '''
    state s: (adj,node_feature)
    :param adj: none*b*n*n
    :param node_feature: none*1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[1]
    batch_size = tf.compat.v1.shape(adj)[0]
    in_channels = node_feature.get_shape()[-1]

    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [1, edge_dim, in_channels, out_channels],initializer=tf.compat.v1.glorot_uniform_initializer())
        b = tf.compat.v1.get_variable("b", [1, edge_dim, 1, out_channels])
        # node_embedding = adj@tf.compat.v1.tile(node_feature,[1,edge_dim,1,1])@tf.compat.v1.tile(W,[batch_size,1,1,1])+b # todo: tf.compat.v1.tile sum the gradients, may need to change
        node_embedding = adj@tf.compat.v1.tile(node_feature,[1,edge_dim,1,1])@tf.compat.v1.tile(W,[batch_size,1,1,1]) # todo: tf.compat.v1.tile sum the gradients, may need to change
        if is_act:
            node_embedding = tf.compat.v1.nn.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = tf.compat.v1.reduce_sum(node_embedding, axis=1, keepdims=True)  # mean pooling
        elif aggregate=='mean':
            node_embedding = tf.compat.v1.reduce_mean(node_embedding,axis=1,keepdims=True) # mean pooling
        elif aggregate=='concat':
            node_embedding = tf.compat.v1.concat(tf.compat.v1.split(node_embedding,axis=1,num_or_size_splits=edge_dim),axis=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = tf.compat.v1.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

# # gcn mean aggregation over edge features, multi hop version
# def GCN_multihop_batch(adj, node_feature, out_channels, hops, is_act=True, is_normalize=False, name='gcn_simple',aggregate='mean'):
#     '''
#     state s: (adj,node_feature)
#     :param adj: none*b*n*n
#     :param node_feature: none*1*n*d
#     :param out_channels: scalar
#     :param name:
#     :return:
#     '''
#     edge_dim = adj.get_shape()[1]
#     batch_size = tf.compat.v1.shape(adj)[0]
#     in_channels = node_feature.get_shape()[-1]
#
#     with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
#         node_embedding_list = []
#         for i in range(hops):
#             W = tf.compat.v1.get_variable("W"+str(i), [1, edge_dim, in_channels, out_channels],initializer=tf.compat.v1.glorot_uniform_initializer())
#             b = tf.compat.v1.get_variable("b"+str(i), [1, edge_dim, 1, out_channels])
#             node_embedding = adj@tf.compat.v1.tile(node_feature,[1,edge_dim,1,1])@tf.compat.v1.tile(W,[batch_size,1,1,1])+b # todo: tf.compat.v1.tile sum the gradients, may need to change
#             if is_act:
#                 node_embedding = tf.compat.v1.nn.relu(node_embedding)
#             if aggregate=='mean':
#                 node_embedding = tf.compat.v1.reduce_mean(node_embedding,axis=1,keepdims=True) # mean pooling
#             elif aggregate=='concat':
#                 node_embedding = tf.compat.v1.concat(tf.compat.v1.split(node_embedding,axis=1,num_or_size_splits=edge_dim),axis=3)
#             else:
#                 print('GCN aggregate error!')
#             if is_normalize:
#                 node_embedding = tf.compat.v1.nn.l2_normalize(node_embedding,axis=-1)
#             node_embedding_list.append(node_embedding)
#         node_embedding = tf.compat.v1.concat(node_embedding_list,axis=-1)
#         return node_embedding

def bilinear(emb_1, emb_2, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    batch_size = tf.compat.v1.shape(emb_1)[0]
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [1, node_dim, node_dim])
        return emb_1 @ tf.compat.v1.tile(W,[batch_size,1,1]) @ tf.compat.v1.transpose(emb_2,[0,2,1])

def bilinear_multi(emb_1, emb_2, out_dim, name='bilinear'):
    node_dim = emb_1.get_shape()[-1]
    batch_size = tf.compat.v1.shape(emb_1)[0]
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [1,out_dim, node_dim, node_dim])
        emb_1 = tf.compat.v1.tile(tf.compat.v1.expand_dims(emb_1,axis=1),[1,out_dim,1,1])
        emb_2 = tf.compat.v1.transpose(emb_2,[0,2,1])
        emb_2 = tf.compat.v1.tile(tf.compat.v1.expand_dims(emb_2,axis=1),[1,out_dim,1,1])
        return emb_1 @ tf.compat.v1.tile(W,[batch_size,1,1,1]) @ emb_2

def emb_node(ob_node,out_channels):
    batch_size = tf.compat.v1.shape(ob_node)[0]
    in_channels = ob_node.get_shape()[-1]
    emb = tf.compat.v1.get_variable('emb',[1,1,in_channels,out_channels])
    return ob_node @ tf.compat.v1.tile(emb,[batch_size,1,1,1])


def discriminator_net(ob,args,name='d_net'):
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        ob_node = tf.compat.v1.layers.dense(ob['node'], 8, activation=None, use_bias=False, name='emb')  # embedding layer
        if args.bn==1:
            ob_node = tf.compat.v1.layers.batch_normalization(ob_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        for i in range(args.layer_num_d - 2):
            if args.bn==1:
                emb_node = tf.compat.v1.layers.batch_normalization(emb_node,axis=-1)
            emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.compat.v1.layers.batch_normalization(emb_node,axis=-1)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        if args.bn==1:
            emb_node = tf.compat.v1.layers.batch_normalization(emb_node,axis=-1)
        # emb_graph = tf.compat.v1.reduce_max(tf.compat.v1.squeeze(emb_node2, axis=1),axis=1)  # B*f
        emb_node = tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, use_bias=False, name='linear1')
        if args.bn==1:
            emb_node = tf.compat.v1.layers.batch_normalization(emb_node,axis=-1)


        if args.gate_sum_d==1:
            emb_node_gate = tf.compat.v1.layers.dense(emb_node,1,activation=tf.compat.v1.nn.sigmoid,name='gate')
            emb_graph = tf.compat.v1.reduce_sum(tf.compat.v1.squeeze(emb_node*emb_node_gate, axis=1),axis=1)  # B*f
        else:
            emb_graph = tf.compat.v1.reduce_sum(tf.compat.v1.squeeze(emb_node, axis=1), axis=1)  # B*f
        logit = tf.compat.v1.layers.dense(emb_graph, 1, activation=None, name='linear2')
        pred = tf.compat.v1.sigmoid(logit)
        # pred = tf.compat.v1.layers.dense(emb_graph, 1, activation=None, name='linear1')
        return pred,logit

def discriminator(x,x_gen,args,name='d_net'):
    d,_ = discriminator_net(x,args,name=name)
    d_,_ = discriminator_net(x_gen,args,name=name)

    d = tf.compat.v1.reduce_mean(d)
    d_ = tf.compat.v1.reduce_mean(d_)
    d_loss = d - d_
    # todo: try adding d_grad_loss

    return d_loss, d, d_


class GCNPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space,args, kind='small', atom_type_num = None):
        with tf.compat.v1.variable_scope(name):
            self._init(ob_space, ac_space, kind, atom_type_num,args)
            self.scope = tf.compat.v1.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind, atom_type_num,args):
        self.pdtype = MultiCatCategoricalPdType
        ### 0 Get input
        ob = {'adj': U.get_placeholder(name="adj", dtype=tf.compat.v1.float32, shape=[None,ob_space['adj'].shape[0],None,None]),
              'node': U.get_placeholder(name="node", dtype=tf.compat.v1.float32, shape=[None,1,None,ob_space['node'].shape[2]])}
        # only when evaluating given action, at training time
        self.ac_real = U.get_placeholder(name='ac_real', dtype=tf.compat.v1.int64, shape=[None,4]) # feed groudtruth action
        ob_node = tf.compat.v1.layers.dense(ob['node'],8,activation=None,use_bias=False,name='emb') # embedding layer
        if args.bn==1:
            ob_node = tf.compat.v1.layers.batch_normalization(ob_node,axis=-1)
        if args.has_concat==1:
            emb_node = tf.compat.v1.concat((GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate),ob_node),axis=-1)
        else:
            emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        if args.bn == 1:
            emb_node = tf.compat.v1.layers.batch_normalization(emb_node, axis=-1)
        for i in range(args.layer_num_g-2):
            if args.has_residual==1:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)+self.emb_node1
            elif args.has_concat==1:
                emb_node = tf.compat.v1.concat((GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate),self.emb_node1),axis=-1)
            else:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_' + str(i + 1),aggregate=args.gcn_aggregate)
            if args.bn == 1:
                emb_node = tf.compat.v1.layers.batch_normalization(emb_node, axis=-1)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        emb_node = tf.compat.v1.squeeze(emb_node,axis=1)  # B*n*f

        ### 1 only keep effective nodes
        # ob_mask = tf.compat.v1.cast(tf.compat.v1.transpose(tf.compat.v1.reduce_sum(ob['node'],axis=-1),[0,2,1]),dtype=tf.compat.v1.bool) # B*n*1
        ob_len = tf.compat.v1.reduce_sum(tf.compat.v1.squeeze(tf.compat.v1.cast(tf.compat.v1.cast(tf.compat.v1.reduce_sum(ob['node'], axis=-1),dtype=tf.compat.v1.bool),dtype=tf.compat.v1.float32),axis=-2),axis=-1)  # B
        ob_len_first = ob_len-atom_type_num
        logits_mask = tf.compat.v1.sequence_mask(ob_len, maxlen=tf.compat.v1.shape(ob['node'])[2]) # mask all valid entry
        logits_first_mask = tf.compat.v1.sequence_mask(ob_len_first,maxlen=tf.compat.v1.shape(ob['node'])[2]) # mask valid entry -3 (rm isolated nodes)

        if args.mask_null==1:
            emb_node_null = tf.compat.v1.zeros(tf.compat.v1.shape(emb_node))
            emb_node = tf.compat.v1.where(condition=tf.compat.v1.tile(tf.compat.v1.expand_dims(logits_mask,axis=-1),(1,1,emb_node.get_shape()[-1])), x=emb_node, y=emb_node_null)

        ## get graph embedding
        emb_graph = tf.compat.v1.reduce_sum(emb_node, axis=1, keepdims=True)
        if args.graph_emb == 1:
            emb_graph = tf.compat.v1.tile(emb_graph, [1, tf.compat.v1.shape(emb_node)[1], 1])
            emb_node = tf.compat.v1.concat([emb_node, emb_graph], axis=2)

        ### 2 predict stop
        emb_stop = tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, use_bias=False, name='linear_stop1')
        if args.bn==1:
            emb_stop = tf.compat.v1.layers.batch_normalization(emb_stop,axis=-1)
        self.logits_stop = tf.compat.v1.reduce_sum(emb_stop,axis=1)
        self.logits_stop = tf.compat.v1.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2_1')  # B*2
        # explicitly show node num
        # self.logits_stop = tf.compat.v1.concat((tf.compat.v1.reduce_mean(tf.compat.v1.layers.dense(emb_node, 32, activation=tf.compat.v1.nn.relu, name='linear_stop1'),axis=1),tf.compat.v1.reshape(ob_len_first/5,[-1,1])),axis=1)
        # self.logits_stop = tf.compat.v1.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2')  # B*2

        stop_shift = tf.compat.v1.constant([[0,args.stop_shift]],dtype=tf.compat.v1.float32)
        pd_stop = CategoricalPdType(-1).pdfromflat(flat=self.logits_stop+stop_shift)
        ac_stop = pd_stop.sample()

        ### 3.1: select first (active) node
        # rules: only select effective nodes
        self.logits_first = tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, name='linear_select1')
        self.logits_first = tf.compat.v1.squeeze(tf.compat.v1.layers.dense(self.logits_first, 1, activation=None, name='linear_select2'),axis=-1) # B*n
        logits_first_null = tf.compat.v1.ones(tf.compat.v1.shape(self.logits_first))*-1000
        self.logits_first = tf.compat.v1.where(condition=logits_first_mask,x=self.logits_first,y=logits_first_null)
        # using own prediction
        pd_first = CategoricalPdType(-1).pdfromflat(flat=self.logits_first)
        ac_first = pd_first.sample()
        mask = tf.compat.v1.one_hot(ac_first, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
        emb_first = tf.compat.v1.boolean_mask(emb_node, mask)
        emb_first = tf.compat.v1.expand_dims(emb_first,axis=1)
        # using groud truth action
        ac_first_real = self.ac_real[:, 0]
        mask_real = tf.compat.v1.one_hot(ac_first_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
        emb_first_real = tf.compat.v1.boolean_mask(emb_node, mask_real)
        emb_first_real = tf.compat.v1.expand_dims(emb_first_real, axis=1)

        ### 3.2: select second node
        # rules: do not select first node
        # using own prediction

        # mlp
        emb_cat = tf.compat.v1.concat([tf.compat.v1.tile(emb_first,[1,tf.compat.v1.shape(emb_node)[1],1]),emb_node],axis=2)
        self.logits_second = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_second1')
        self.logits_second = tf.compat.v1.layers.dense(self.logits_second, 1, activation=None, name='logits_second2')
        # # bilinear
        # self.logits_second = tf.compat.v1.transpose(bilinear(emb_first, emb_node, name='logits_second'), [0, 2, 1])

        self.logits_second = tf.compat.v1.squeeze(self.logits_second, axis=-1)
        ac_first_mask = tf.compat.v1.one_hot(ac_first, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=False, off_value=True)
        logits_second_mask = tf.compat.v1.logical_and(logits_mask,ac_first_mask)
        logits_second_null = tf.compat.v1.ones(tf.compat.v1.shape(self.logits_second)) * -1000
        self.logits_second = tf.compat.v1.where(condition=logits_second_mask, x=self.logits_second, y=logits_second_null)

        pd_second = CategoricalPdType(-1).pdfromflat(flat=self.logits_second)
        ac_second = pd_second.sample()
        mask = tf.compat.v1.one_hot(ac_second, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
        emb_second = tf.compat.v1.boolean_mask(emb_node, mask)
        emb_second = tf.compat.v1.expand_dims(emb_second, axis=1)

        # using groudtruth
        # mlp
        emb_cat = tf.compat.v1.concat([tf.compat.v1.tile(emb_first_real, [1, tf.compat.v1.shape(emb_node)[1], 1]), emb_node], axis=2)
        self.logits_second_real = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_second1',reuse=True)
        self.logits_second_real = tf.compat.v1.layers.dense(self.logits_second_real, 1, activation=None, name='logits_second2',reuse=True)
        # # bilinear
        # self.logits_second_real = tf.compat.v1.transpose(bilinear(emb_first_real, emb_node, name='logits_second'), [0, 2, 1])

        self.logits_second_real = tf.compat.v1.squeeze(self.logits_second_real, axis=-1)
        ac_first_mask_real = tf.compat.v1.one_hot(ac_first_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=False, off_value=True)
        logits_second_mask_real = tf.compat.v1.logical_and(logits_mask,ac_first_mask_real)
        self.logits_second_real = tf.compat.v1.where(condition=logits_second_mask_real, x=self.logits_second_real, y=logits_second_null)

        ac_second_real = self.ac_real[:,1]
        mask_real = tf.compat.v1.one_hot(ac_second_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
        emb_second_real = tf.compat.v1.boolean_mask(emb_node, mask_real)
        emb_second_real = tf.compat.v1.expand_dims(emb_second_real, axis=1)

        ### 3.3 predict edge type
        # using own prediction
        # MLP
        emb_cat = tf.compat.v1.concat([emb_first,emb_second],axis=-1)
        self.logits_edge = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_edge1')
        self.logits_edge = tf.compat.v1.layers.dense(self.logits_edge, ob['adj'].get_shape()[1], activation=None, name='logits_edge2')
        self.logits_edge = tf.compat.v1.squeeze(self.logits_edge,axis=1)
        # # bilinear
        # self.logits_edge = tf.compat.v1.reshape(bilinear_multi(emb_first,emb_second,out_dim=ob['adj'].get_shape()[1]),[-1,ob['adj'].get_shape()[1]])
        pd_edge = CategoricalPdType(-1).pdfromflat(self.logits_edge)
        ac_edge = pd_edge.sample()

        # using ground truth
        # MLP
        emb_cat = tf.compat.v1.concat([emb_first_real, emb_second_real], axis=-1)
        self.logits_edge_real = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_edge1', reuse=True)
        self.logits_edge_real = tf.compat.v1.layers.dense(self.logits_edge_real, ob['adj'].get_shape()[1], activation=None,
                                           name='logits_edge2', reuse=True)
        self.logits_edge_real = tf.compat.v1.squeeze(self.logits_edge_real, axis=1)
        # # bilinear
        # self.logits_edge_real = tf.compat.v1.reshape(bilinear_multi(emb_first_real, emb_second_real, out_dim=ob['adj'].get_shape()[1]),
        #                               [-1, ob['adj'].get_shape()[1]])


        # ncat_list = [tf.compat.v1.shape(logits_first),ob_space['adj'].shape[-1],ob_space['adj'].shape[0]]
        self.pd = self.pdtype(-1).pdfromflat([self.logits_first,self.logits_second_real,self.logits_edge_real,self.logits_stop])
        self.vpred = tf.compat.v1.layers.dense(emb_node, args.emb_size, use_bias=False, activation=tf.compat.v1.nn.relu, name='value1')
        if args.bn==1:
            self.vpred = tf.compat.v1.layers.batch_normalization(self.vpred,axis=-1)
        self.vpred = tf.compat.v1.reduce_max(self.vpred,axis=1)
        self.vpred = tf.compat.v1.layers.dense(self.vpred, 1, activation=None, name='value2')

        self.state_in = []
        self.state_out = []

        self.ac = tf.compat.v1.concat((tf.compat.v1.expand_dims(ac_first,axis=1),tf.compat.v1.expand_dims(ac_second,axis=1),tf.compat.v1.expand_dims(ac_edge,axis=1),tf.compat.v1.expand_dims(ac_stop,axis=1)),axis=1)


        debug = {}
        debug['ob_node'] = tf.compat.v1.shape(ob['node'])
        debug['ob_adj'] = tf.compat.v1.shape(ob['adj'])
        debug['emb_node'] = emb_node
        debug['logits_stop'] = self.logits_stop
        debug['logits_second'] = self.logits_second
        debug['ob_len'] = ob_len
        debug['logits_first_mask'] = logits_first_mask
        debug['logits_second_mask'] = logits_second_mask
        # debug['pd'] = self.pd.logp(self.ac)
        debug['ac'] = self.ac

        stochastic = tf.compat.v1.placeholder(dtype=tf.compat.v1.bool, shape=())
        self._act = U.function([stochastic, ob['adj'], ob['node']], [self.ac, self.vpred, debug]) # add debug in second arg if needed

    def act(self, stochastic, ob):
        return self._act(stochastic, ob['adj'][None], ob['node'][None])
        # return self._act(stochastic, ob['adj'], ob['node'])

    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []








def GCN_emb(ob,args):
    ob_node = tf.compat.v1.layers.dense(ob['node'], 8, activation=None, use_bias=False, name='emb')  # embedding layer
    if args.has_concat == 1:
        emb_node1 = tf.compat.v1.concat(
            (GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1', aggregate=args.gcn_aggregate), ob_node),
            axis=-1)
    else:
        emb_node1 = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1', aggregate=args.gcn_aggregate)
    for i in range(args.layer_num_g - 2):
        if args.has_residual == 1:
            emb_node1 = GCN_batch(ob['adj'], emb_node1, args.emb_size, name='gcn1_' + str(i + 1),
                                       aggregate=args.gcn_aggregate) + emb_node1
        elif args.has_concat == 1:
            emb_node1 = tf.compat.v1.concat((GCN_batch(ob['adj'], emb_node1, args.emb_size,
                                                  name='gcn1_' + str(i + 1), aggregate=args.gcn_aggregate),
                                        emb_node1), axis=-1)
        else:
            emb_node1 = GCN_batch(ob['adj'], emb_node1, args.emb_size, name='gcn1_' + str(i + 1),
                                       aggregate=args.gcn_aggregate)
    emb_node2 = GCN_batch(ob['adj'], emb_node1, args.emb_size, is_act=False, is_normalize=True,
                               name='gcn2', aggregate=args.gcn_aggregate)
    emb_node = tf.compat.v1.squeeze(emb_node2, axis=1)  # B*n*f
    emb_graph = tf.compat.v1.reduce_max(emb_node, axis=1, keepdims=True)
    if args.graph_emb == 1:
        emb_graph = tf.compat.v1.tile(emb_graph, [1, tf.compat.v1.shape(emb_node)[1], 1])
        emb_node = tf.compat.v1.concat([emb_node, emb_graph], axis=2)
    return emb_node


#### debug

if __name__ == "__main__":
    adj_np = np.ones((5,3,4,4))
    adj = tf.compat.v1.placeholder(shape=(5,3,4,4),dtype=tf.compat.v1.float32)
    node_feature_np = np.ones((5,1,4,3))
    node_feature = tf.compat.v1.placeholder(shape=(5,1,4,3),dtype=tf.compat.v1.float32)


    ob_space = {}
    atom_type = 5
    ob_space['adj'] = gym.Space(shape=[3,5,5])
    ob_space['node'] = gym.Space(shape=[1,5,atom_type])
    ac_space = gym.spaces.MultiDiscrete([10, 10, 3])
    policy = GCNPolicy(name='policy',ob_space=ob_space,ac_space=ac_space)

    stochastic = True
    env = gym.make('molecule-v0')  # in gym format
    env.init()
    ob = env.reset()

    # ob['adj'] = np.repeat(ob['adj'][None],2,axis=0)
    # ob['node'] = np.repeat(ob['node'][None],2,axis=0)

    print('adj',ob['adj'].shape)
    print('node',ob['node'].shape)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(20):
            ob = env.reset()
            for j in range(0,20):
                ac,vpred,debug = policy.act(stochastic,ob)
                # if ac[0]==ac[1]:
                #     print('error')
                # else:
                # print('i',i,'ac',ac,'vpred',vpred,'debug',debug['logits_first'].shape,debug['logits_second'].shape)
                print('i', i)
                # print('ac\n',ac)
                # print('debug\n',debug['ob_len'])
                ob,reward,_,_ = env.step(ac)








# class GCNPolicy_scaffold(object):
#     recurrent = False
#     def __init__(self, name, ob_space, ac_space,args, kind='small', atom_type_num = None):
#         with tf.compat.v1.variable_scope(name):
#             self._init(ob_space, ac_space, kind, atom_type_num,args)
#             self.scope = tf.compat.v1.get_variable_scope().name
#
#     def _init(self, ob_space, ac_space, kind, atom_type_num,args):
#         self.pdtype = MultiCatCategoricalPdType
#         ### 0 Get input
#         ob = {'adj': U.get_placeholder(name="adj", dtype=tf.compat.v1.float32, shape=[None,ob_space['adj'].shape[0],None,None]),
#               'node': U.get_placeholder(name="node", dtype=tf.compat.v1.float32, shape=[None,1,None,ob_space['node'].shape[2]])}
#         ob_scaffold = {
#             'adj': U.get_placeholder(name="adj_scaffold", dtype=tf.compat.v1.float32, shape=[None, ob_space['adj'].shape[0], None, None]),
#             'node': U.get_placeholder(name="node_scaffold", dtype=tf.compat.v1.float32, shape=[None, 1, None, ob_space['node'].shape[2]])}
#
#         # only when evaluating given action, at training time
#         self.ac_real = U.get_placeholder(name='ac_real', dtype=tf.compat.v1.int64, shape=[None,4]) # feed groudtruth action
#         if kind == 'small':
#             emb_node = GCN_emb(ob,args)
#             emb_scaffold = GCN_emb(ob_scaffold,args)
#
#         else:
#             raise NotImplementedError
#
#         ### 1 only keep effective nodes
#         # ob_mask = tf.compat.v1.cast(tf.compat.v1.transpose(tf.compat.v1.reduce_sum(ob['node'],axis=-1),[0,2,1]),dtype=tf.compat.v1.bool) # B*n*1
#         ob_len = tf.compat.v1.reduce_sum(tf.compat.v1.squeeze(tf.compat.v1.reduce_sum(ob['node'], axis=-1),axis=-2),axis=-1)  # B
#         ob_len_first = ob_len-atom_type_num
#         logits_mask = tf.compat.v1.sequence_mask(ob_len, maxlen=tf.compat.v1.shape(ob['node'])[2]) # mask all valid entry
#         logits_first_mask = tf.compat.v1.sequence_mask(ob_len_first,maxlen=tf.compat.v1.shape(ob['node'])[2]) # mask valid entry -3 (rm isolated nodes)
#
#         ## for scaffold
#         # get scaffold emb
#         batch_size = tf.compat.v1.shape(emb_node)[0]
#         emb_scaffold = tf.compat.v1.tile(tf.compat.v1.reshape(emb_scaffold,[1, -1, emb_scaffold.get_shape()[-1]]), [batch_size, 1, 1])  # B * B_s*6 *F
#
#         ob_len_scaffold = tf.compat.v1.reduce_sum(tf.compat.v1.squeeze(tf.compat.v1.reduce_sum(ob_scaffold['node'], axis=-1), axis=-2), axis=-1)  # B
#         logits_scaffold_mask = tf.compat.v1.tile(tf.compat.v1.reshape(tf.compat.v1.sequence_mask(ob_len_scaffold, maxlen=tf.compat.v1.shape(ob_scaffold['node'])[2]),[1,-1]),[batch_size,1])   # mask all valid entry
#
#         ### 2 predict stop
#         self.logits_stop = tf.compat.v1.reduce_sum(tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, name='linear_stop1'),axis=1)
#         self.logits_stop = tf.compat.v1.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2_1')  # B*2
#         # explicitly show node num
#         # self.logits_stop = tf.compat.v1.concat((tf.compat.v1.reduce_mean(tf.compat.v1.layers.dense(emb_node, 32, activation=tf.compat.v1.nn.relu, name='linear_stop1'),axis=1),tf.compat.v1.reshape(ob_len_first/5,[-1,1])),axis=1)
#         # self.logits_stop = tf.compat.v1.layers.dense(self.logits_stop, 2, activation=None, name='linear_stop2')  # B*2
#
#         stop_shift = tf.compat.v1.constant([[0,args.stop_shift]],dtype=tf.compat.v1.float32)
#         pd_stop = CategoricalPdType(-1).pdfromflat(flat=self.logits_stop+stop_shift)
#         ac_stop = pd_stop.sample()
#
#         ### 3.1: select first (active) node
#         # rules: only select effective nodes
#         self.logits_first = tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, name='linear_select1')
#         self.logits_first = tf.compat.v1.squeeze(tf.compat.v1.layers.dense(self.logits_first, 1, activation=None, name='linear_select2'),axis=-1) # B*n
#         logits_first_null = tf.compat.v1.ones(tf.compat.v1.shape(self.logits_first))*-1000
#         self.logits_first = tf.compat.v1.where(condition=logits_first_mask,x=self.logits_first,y=logits_first_null)
#         # using own prediction
#         pd_first = CategoricalPdType(-1).pdfromflat(flat=self.logits_first)
#         ac_first = pd_first.sample()
#         mask = tf.compat.v1.one_hot(ac_first, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
#         emb_first = tf.compat.v1.boolean_mask(emb_node, mask)
#         emb_first = tf.compat.v1.expand_dims(emb_first,axis=1)
#         # using groud truth action
#         ac_first_real = self.ac_real[:, 0]
#         mask_real = tf.compat.v1.one_hot(ac_first_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
#         emb_first_real = tf.compat.v1.boolean_mask(emb_node, mask_real)
#         emb_first_real = tf.compat.v1.expand_dims(emb_first_real, axis=1)
#
#         ### 3.2: select second node
#         # rules: do not select first node
#         # concat emb_node with emb_scaffold
#         emb_node = tf.compat.v1.concat((emb_node,emb_scaffold),axis=-1)
#
#         # using own prediction
#         # mlp
#         emb_cat = tf.compat.v1.concat([tf.compat.v1.tile(emb_first,[1,tf.compat.v1.shape(emb_node)[1],1]),emb_node],axis=2)
#         self.logits_second = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_second1')
#         self.logits_second = tf.compat.v1.layers.dense(self.logits_second, 1, activation=None, name='logits_second2')
#         # # bilinear
#         # self.logits_second = tf.compat.v1.transpose(bilinear(emb_first, emb_node, name='logits_second'), [0, 2, 1])
#
#         self.logits_second = tf.compat.v1.squeeze(self.logits_second, axis=-1)
#         ac_first_mask = tf.compat.v1.one_hot(ac_first, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=False, off_value=True)
#         logits_second_mask = tf.compat.v1.logical_and(tf.compat.v1.concat((logits_mask,logits_scaffold_mask),axis=-1),ac_first_mask)
#         logits_second_null = tf.compat.v1.ones(tf.compat.v1.shape(self.logits_second)) * -1000
#         self.logits_second = tf.compat.v1.where(condition=logits_second_mask, x=self.logits_second, y=logits_second_null)
#
#         pd_second = CategoricalPdType(-1).pdfromflat(flat=self.logits_second)
#         ac_second = pd_second.sample()
#         mask = tf.compat.v1.one_hot(ac_second, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
#         emb_second = tf.compat.v1.boolean_mask(emb_node, mask)
#         emb_second = tf.compat.v1.expand_dims(emb_second, axis=1)
#
#         # using groudtruth
#         # mlp
#         emb_cat = tf.compat.v1.concat([tf.compat.v1.tile(emb_first_real, [1, tf.compat.v1.shape(emb_node)[1], 1]), emb_node], axis=2)
#         self.logits_second_real = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_second1',reuse=True)
#         self.logits_second_real = tf.compat.v1.layers.dense(self.logits_second_real, 1, activation=None, name='logits_second2',reuse=True)
#         # # bilinear
#         # self.logits_second_real = tf.compat.v1.transpose(bilinear(emb_first_real, emb_node, name='logits_second'), [0, 2, 1])
#
#         self.logits_second_real = tf.compat.v1.squeeze(self.logits_second_real, axis=-1)
#         ac_first_mask_real = tf.compat.v1.one_hot(ac_first_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=False, off_value=True)
#         logits_second_mask_real = tf.compat.v1.logical_and(tf.compat.v1.concat((logits_mask,logits_scaffold_mask),axis=-1),ac_first_mask_real)
#         self.logits_second_real = tf.compat.v1.where(condition=logits_second_mask_real, x=self.logits_second_real, y=logits_second_null)
#
#         ac_second_real = self.ac_real[:,1]
#         mask_real = tf.compat.v1.one_hot(ac_second_real, depth=tf.compat.v1.shape(emb_node)[1], dtype=tf.compat.v1.bool, on_value=True, off_value=False)
#         emb_second_real = tf.compat.v1.boolean_mask(emb_node, mask_real)
#         emb_second_real = tf.compat.v1.expand_dims(emb_second_real, axis=1)
#
#         ### 3.3 predict edge type
#         # using own prediction
#         # MLP
#         emb_cat = tf.compat.v1.concat([emb_first,emb_second],axis=-1)
#         self.logits_edge = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_edge1')
#         self.logits_edge = tf.compat.v1.layers.dense(self.logits_edge, ob['adj'].get_shape()[1], activation=None, name='logits_edge2')
#         self.logits_edge = tf.compat.v1.squeeze(self.logits_edge,axis=1)
#         # # bilinear
#         # self.logits_edge = tf.compat.v1.reshape(bilinear_multi(emb_first,emb_second,out_dim=ob['adj'].get_shape()[1]),[-1,ob['adj'].get_shape()[1]])
#         pd_edge = CategoricalPdType(-1).pdfromflat(self.logits_edge)
#         ac_edge = pd_edge.sample()
#
#         # using ground truth
#         # MLP
#         emb_cat = tf.compat.v1.concat([emb_first_real, emb_second_real], axis=-1)
#         self.logits_edge_real = tf.compat.v1.layers.dense(emb_cat, args.emb_size, activation=tf.compat.v1.nn.relu, name='logits_edge1', reuse=True)
#         self.logits_edge_real = tf.compat.v1.layers.dense(self.logits_edge_real, ob['adj'].get_shape()[1], activation=None,
#                                            name='logits_edge2', reuse=True)
#         self.logits_edge_real = tf.compat.v1.squeeze(self.logits_edge_real, axis=1)
#         # # bilinear
#         # self.logits_edge_real = tf.compat.v1.reshape(bilinear_multi(emb_first_real, emb_second_real, out_dim=ob['adj'].get_shape()[1]),
#         #                               [-1, ob['adj'].get_shape()[1]])
#
#
#         # ncat_list = [tf.compat.v1.shape(logits_first),ob_space['adj'].shape[-1],ob_space['adj'].shape[0]]
#         self.pd = self.pdtype(-1).pdfromflat([self.logits_first,self.logits_second_real,self.logits_edge_real,self.logits_stop])
#         self.vpred = tf.compat.v1.layers.dense(emb_node, args.emb_size, activation=tf.compat.v1.nn.relu, name='value1')
#         self.vpred = tf.compat.v1.layers.dense(self.vpred, 1, activation=None, name='value2')
#         self.vpred = tf.compat.v1.reduce_max(self.vpred,axis=1)
#
#         self.state_in = []
#         self.state_out = []
#
#         self.ac = tf.compat.v1.concat((tf.compat.v1.expand_dims(ac_first,axis=1),tf.compat.v1.expand_dims(ac_second,axis=1),tf.compat.v1.expand_dims(ac_edge,axis=1),tf.compat.v1.expand_dims(ac_stop,axis=1)),axis=1)
#
#         # print('ob_adj', ob['adj'].get_shape(),
#         #       'ob_node', ob['node'].get_shape())
#         # print('logits_first', self.logits_first.get_shape(),
#         #       'logits_second', self.logits_second.get_shape(),
#         #       'logits_edge', self.logits_edge.get_shape())
#         # print('ac_edge', ac_edge.get_shape())
#         # for var in tf.compat.v1.trainable_variables():
#         #     print('variable', var)
#
#         debug = {}
#         # debug['ob_node'] = tf.compat.v1.shape(ob['node'])
#         # debug['ob_adj'] = tf.compat.v1.shape(ob['adj'])
#         # debug['emb_node'] = emb_node
#         # debug['emb_node1'] = self.emb_node1
#         # debug['emb_node2'] = self.emb_node2
#         # debug['logits_stop'] = self.logits_stop
#         # debug['logits_second'] = self.logits_second
#         # debug['ob_len'] = ob_len
#         # debug['logits_first_mask'] = logits_first_mask
#         # debug['logits_second_mask'] = logits_second_mask
#         # # debug['pd'] = self.pd.logp(self.ac)
#         # debug['ac'] = self.ac
#
#         stochastic = tf.compat.v1.placeholder(dtype=tf.compat.v1.bool, shape=())
#         self._act = U.function([stochastic, ob['adj'], ob['node'],ob_scaffold['adj'],ob_scaffold['node']], [self.ac, self.vpred, debug]) # add debug in second arg if needed
#
#     def act(self, stochastic, ob):
#         return self._act(stochastic, ob['adj'][None], ob['node'][None], ob['adj_scaffold'][None], ob['node_scaffold'][None])
#         # return self._act(stochastic, ob['adj'], ob['node'])
#
#     def get_variables(self):
#         return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)
#     def get_trainable_variables(self):
#         return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
#     def get_initial_state(self):
#         return []
#
#
#
#
