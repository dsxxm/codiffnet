from __future__ import division
import tensorflow as tf
import numpy as np

class diffnetplus():
    def __init__(self, conf):
        self.conf = conf
        self.conf.dimension = self.parseDimonsion(conf)
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX',
            'ITEM_ITEM_SPARSE_MATRIX',  # MODIFIED: GraphTrust 物品-物品图
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()


    def inputSupply(self, data_dict):
        low_att_std = 1.0

        ########  Node Attention initialization ########

        # ----------------------
        # user-user social network node attention initialization
        self.first_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_SN_layer1')
        self.first_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_SN_layer2')

        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.social_neighbors_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=low_att_std)),[-1,1])      )   ),1)

        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_values_input1, axes=0)
        self.first_user_user_low_att = [first_mean_social_influ, first_var_social_influ]

        self.second_low_att_layer_for_social_neighbors_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_SN_layer1')
        self.second_low_att_layer_for_social_neighbors_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_SN_layer2')
        self.social_neighbors_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_social_neighbors_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=1.0)),[-1,1])      )   ),1)

        self.social_neighbors_values_input3 = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_input)], stddev=0.01))
        self.social_neighbors_num_input = 1.0/np.reshape(data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'],[-1,1])

        # ----------------------
        # user-item interest graph node attention initialization
        self.first_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_UI_layer1')
        self.first_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_UI_layer2')

        self.user_item_sparsity_dict = data_dict['USER_ITEM_SPARSITY_DICT']
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']
        #self.consumed_items_values_input1 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=low_att_std)),[-1,1])  )   ),1)
        
        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.consumed_items_values_input1, axes=0)
        self.first_user_item_low_att = [first_mean_social_influ, first_var_social_influ]


        self.second_low_att_layer_for_user_item_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_UI_layer1')
        self.second_low_att_layer_for_user_item_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_UI_layer2')
        #self.consumed_items_values_input2 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_user_item_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=1.0)),[-1,1])    )   ),1)

        self.consumed_items_values_input3 = tf.Variable(tf.random_normal([len(self.consumed_items_indices_input)], stddev=0.01))
        self.consumed_items_num_input = 1.0/np.reshape(data_dict['CONSUMED_ITEMS_NUM_INPUT'], [-1,1])


        # ----------------------
        # item-user graph node attention initialization
        self.first_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='first_low_att_IU_layer1')
        self.first_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='first_low_att_IU_layer2')


        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']
        #self.item_customer_values_input1 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input1 = tf.reduce_sum(tf.math.exp(self.first_low_att_layer_for_item_user_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=low_att_std)),[-1,1])    )   ),1)


        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.item_customer_values_input1, axes=0)
        self.first_item_user_low_att = [first_mean_social_influ, first_var_social_influ]


        self.second_low_att_layer_for_item_user_layer1 = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='second_low_att_IU_layer1')
        self.second_low_att_layer_for_item_user_layer2 = tf.layers.Dense(1, activation=tf.nn.leaky_relu, name='second_low_att_IU_layer2')
        #self.item_customer_values_input2 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_values_input2 = tf.reduce_sum(tf.math.exp(self.second_low_att_layer_for_item_user_layer1( \
                                            tf.reshape(tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01)),[-1,1]) )   ),1)

        self.item_customer_values_input3 = tf.Variable(tf.random_normal([len(self.item_customer_indices_input)], stddev=0.01))
        self.item_customer_num_input = 1.0/np.reshape(data_dict['ITEM_CUSTOMER_NUM_INPUT'],[-1,1])

        # MODIFIED: GraphTrust 物品-物品图初始化
        self.item_item_indices_input = data_dict['ITEM_ITEM_INDICES_INPUT']
        self.item_item_values_input = data_dict['ITEM_ITEM_VALUES_INPUT']

        # ----------------------
        # prepare the shape of sparse matrice
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.item_customer_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)
        # MODIFIED: 物品-物品图的形状
        self.item_item_dense_shape = np.array([self.conf.num_items, self.conf.num_items]).astype(np.int64)


        ########  Rough Graph Attention initialization ########
        # ----------------------
        # User part
        # First Layer Influence:
        self.first_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_user_part_influence_attention")
        first_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_user_attention_ini),1)
        self.first_layer_user_attention = tf.div(tf.math.exp(self.first_layer_user_attention_ini), first_layer_user_attention_norm_denominator)
        self.first_user_userneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,0],[1,1])
        self.first_user_itemneighbor_attention_value = tf.slice(self.first_layer_user_attention,[0,1],[1,1])

        # Second Layer Influence:
        self.second_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_user_part_influence_attention")
        second_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_user_attention_ini),1)
        second_layer_user_attention = tf.div(tf.math.exp(self.second_layer_user_attention_ini), second_layer_user_attention_norm_denominator)
        self.second_user_userneighbor_attention_value = tf.slice(second_layer_user_attention,[0,0],[1,1])
        self.second_user_itemneighbor_attention_value = tf.slice(second_layer_user_attention,[0,1],[1,1])        

        # Third Layer Influence:
        self.third_layer_user_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_user_part_influence_attention")
        third_layer_user_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_user_attention_ini),1)
        third_layer_user_attention = tf.div(tf.math.exp(self.third_layer_user_attention_ini), third_layer_user_attention_norm_denominator)
        self.third_user_userneighbor_attention_value = tf.slice(third_layer_user_attention,[0,0],[1,1])
        self.third_user_itemneighbor_attention_value = tf.slice(third_layer_user_attention,[0,1],[1,1])     


        # ----------------------
        # Item part
        # First Layer Influence:
        self.first_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "first_layer_item_part_influence_attention")
        first_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.first_layer_item_attention_ini),1)
        self.first_layer_item_attention = tf.div(tf.math.exp(self.first_layer_item_attention_ini), first_layer_item_attention_norm_denominator)

        self.first_item_itself_attention_value = tf.slice(self.first_layer_item_attention,[0,0],[1,1])
        self.first_item_userneighbor_attention_value = tf.slice(self.first_layer_item_attention,[0,1],[1,1])

        # Second Layer Influence:
        self.second_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "second_layer_item_part_influence_attention")
        second_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.second_layer_item_attention_ini),1)
        second_layer_item_attention = tf.div(tf.math.exp(self.second_layer_item_attention_ini), second_layer_item_attention_norm_denominator)

        self.second_item_itself_attention_value = tf.slice(second_layer_item_attention,[0,0],[1,1])
        self.second_item_userneighbor_attention_value = tf.slice(second_layer_item_attention,[0,1],[1,1])        

        # Third Layer Influence:
        self.third_layer_item_attention_ini = tf.Variable(tf.random_normal([1, 2], stddev=0.01), name = "third_layer_item_part_influence_attention")
        third_layer_item_attention_norm_denominator = tf.reduce_sum(tf.math.exp(self.third_layer_item_attention_ini),1)
        third_layer_item_attention = tf.div(tf.math.exp(self.third_layer_item_attention_ini), third_layer_item_attention_norm_denominator)

        self.third_item_itself_attention_value = tf.slice(third_layer_item_attention,[0,0],[1,1])
        self.third_item_userneighbor_attention_value = tf.slice(third_layer_item_attention,[0,1],[1,1])      


        ######## Generate Sparse Matrices with/without attention #########

        # ----------------------
        # Frist Layer

        self.social_neighbors_sparse_matrix_avg = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input1,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix_avg = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input1,
            dense_shape=self.consumed_items_dense_shape
        )
        self.item_customer_sparse_matrix_avg = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input,
            dense_shape=self.item_customer_dense_shape
        )
        self.first_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input1,
            dense_shape=self.item_customer_dense_shape
        )
        self.first_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.first_layer_social_neighbors_sparse_matrix) 
        self.first_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.first_layer_consumed_items_sparse_matrix) 
        self.first_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.first_layer_item_customer_sparse_matrix) 
        
        # ----------------------
        # Second layer 

        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input2,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input2,
            dense_shape=self.consumed_items_dense_shape
        )
        self.second_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input2,
            dense_shape=self.item_customer_dense_shape
        )

        
        self.second_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.second_layer_social_neighbors_sparse_matrix) 
        self.second_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.second_layer_consumed_items_sparse_matrix) 
        self.second_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.second_layer_item_customer_sparse_matrix) 
        

        # ----------------------
        # Third layer 

        self.third_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input3,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.third_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input3,
            dense_shape=self.consumed_items_dense_shape
        )
        self.third_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices = self.item_customer_indices_input, 
            values = self.item_customer_values_input3,
            dense_shape=self.item_customer_dense_shape
        )

        self.third_social_neighbors_low_level_att_matrix = tf.sparse.softmax(self.third_layer_social_neighbors_sparse_matrix)
        self.third_consumed_items_low_level_att_matrix = tf.sparse.softmax(self.third_layer_consumed_items_sparse_matrix)
        self.third_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(self.third_layer_item_customer_sparse_matrix)

        # MODIFIED: GraphTrust 物品-物品稀疏矩阵 (关系4)
        # 直接使用 DataModule 中已度归一化的值，不做 softmax，避免大矩阵性能开销
        self.first_items_items_low_level_att_matrix = tf.SparseTensor(
            indices=self.item_item_indices_input,
            values=self.item_item_values_input,
            dense_shape=self.item_item_dense_shape
        )


    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y

    def buildGraphContrastiveLoss(self, embeddings, edge_indices, num_nodes, num_samples):
        """图结构感知的对比学习损失

        正样本: 图中真实相连的节点对 (社交边 / 物品共现边)
        负样本: 随机配对的节点对
        损失 = softplus(neg_score - pos_score)  (BPR 风格)

        Args:
            embeddings:   [num_nodes, dim] 嵌入矩阵
            edge_indices: [num_edges, 2] 边索引
            num_nodes:    总节点数
            num_samples:  每次采样的对数
        Returns:
            scalar 对比损失
        """
        # 随机采样正边
        edge_count = tf.shape(edge_indices)[0]
        actual_samples = tf.minimum(num_samples, edge_count)
        sample_idx = tf.random.uniform([actual_samples], maxval=edge_count, dtype=tf.int32)
        pos_edges = tf.gather(edge_indices, sample_idx)  # [S, 2]

        # 随机采样负对
        neg_u = tf.random.uniform([actual_samples], maxval=num_nodes, dtype=tf.int32)
        neg_v = tf.random.uniform([actual_samples], maxval=num_nodes, dtype=tf.int32)

        # L2 归一化
        norm_emb = tf.math.l2_normalize(embeddings, axis=1)

        # 正样本: 点积相似度
        pos_u = tf.gather(norm_emb, pos_edges[:, 0])
        pos_v = tf.gather(norm_emb, pos_edges[:, 1])
        pos_score = tf.reduce_sum(tf.multiply(pos_u, pos_v), axis=1)

        # 负样本: 随机对
        neg_u_emb = tf.gather(norm_emb, neg_u)
        neg_v_emb = tf.gather(norm_emb, neg_v)
        neg_score = tf.reduce_sum(tf.multiply(neg_u_emb, neg_v_emb), axis=1)

        # softplus 损失 = log(1 + exp(neg - pos)), 数值稳定
        loss = tf.reduce_mean(tf.nn.softplus(neg_score - pos_score))

        return loss

    # ----------------------
    # Operations for Diffusion

    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix_avg, current_user_embedding
        )
        return user_embedding_from_social_neighbors


    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.first_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
   

    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix_avg, current_item_embedding
        )
        return user_embedding_from_consumed_items


    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.first_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items


    def generateItemEmebddingFromCustomer(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.item_customer_sparse_matrix_avg, current_user_embedding
        )
        return item_embedding_from_customer


    def generateItemEmebddingFromCustomer1(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.first_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer


    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.second_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    
    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.second_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.second_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    def generateUserEmbeddingFromSocialNeighbors3(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.third_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors
    
    def parseDimonsion(self,conf):
        return conf.dimension*2
    
    def generateUserEmebddingFromConsumedItems3(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.third_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer3(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse_tensor_dense_matmul(
            self.third_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer
    
    # MODIFIED: GraphTrust 物品-物品影响力传播
    def generateItemEmebddingFromRelatedItems(self, current_item_embedding):
        item_embedding_from_related = tf.sparse_tensor_dense_matmul(
            self.first_items_items_low_level_att_matrix, current_item_embedding
        )
        return item_embedding_from_related


    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')

        self.user_review_vector_matrix = tf.constant(\
            np.load(self.conf.user_review_vector_matrix), dtype=tf.float32)
        self.item_review_vector_matrix = tf.constant(\
            np.load(self.conf.item_review_vector_matrix), dtype=tf.float32)
        self.reduce_dimension_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')

        # MODIFIED: 自适应残差门控层
        # 用于替代原始固定的 0.5*self + 0.5*neighbors
        # 让模型根据当前层表示自适应决定“保留自身”与“吸收邻居”的比例
        self.first_user_residual_gate_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='first_user_residual_gate_layer')
        self.second_user_residual_gate_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='second_user_residual_gate_layer')
        self.first_item_residual_gate_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='first_item_residual_gate_layer')
        self.second_item_residual_gate_layer = tf.layers.Dense(\
            self.conf.dimension, activation=tf.nn.sigmoid, name='second_item_residual_gate_layer')

        ########  Fine-grained Graph Attention initialization ########
        # ----------------------
        # User part

        # ----------------------
        # First diffusion layer
        self.first_user_part_social_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_UU_user_MLP_first_layer')
        self.first_user_part_social_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_UU_user_MLP_sencond_layer')
        self.first_user_part_social_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_UU_user_MLP_third_layer')

        self.first_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_UI_user_MLP_first_layer')
        self.first_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_UI_user_MLP_second_layer')
        self.first_user_part_interest_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_UI_user_MLP_third_layer')

        # ----------------------
        # Second diffusion layer
        self.second_user_part_social_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_UU_user_MLP_first_layer')
        self.second_user_part_social_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_UU_user_MLP_second_layer')
        self.second_user_part_social_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_UU_user_MLP_third_layer')

        self.second_user_part_interest_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_UI_user_MLP_first_layer')
        self.second_user_part_interest_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_UI_user_MLP_second_layer')
        self.second_user_part_interest_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_UI_user_MLP_third_layer')

        # ----------------------
        # Item part
        self.first_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_IU_itemself_MLP_first_layer')
        self.first_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_IU_itemself_MLP_second_layer')
        self.first_item_part_itself_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_IU_itemself_MLP_third_layer')

        self.first_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='firstGCN_IU_customer_MLP_first_layer')
        self.first_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_IU_customer_MLP_second_layer')
        self.first_item_part_user_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='firstGCN_IU_customer_MLP_third_layer')

        self.second_item_part_itself_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_IU_itemself_MLP_first_layer')
        self.second_item_part_itself_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_itemself_MLP_second_layer')
        self.second_item_part_itself_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_itemself_MLP_third_layer')

        self.second_item_part_user_graph_att_layer1 = tf.layers.Dense(\
            1, activation=tf.nn.tanh, name='secondGCN_IU_customer_MLP_first_layer')
        self.second_item_part_user_graph_att_layer2 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_customer_MLP_second_layer')
        self.second_item_part_user_graph_att_layer3 = tf.layers.Dense(\
            1, activation=tf.nn.leaky_relu, name='secondGCN_IU_customer_MLP_third_layer')



    def constructTrainGraph(self):

        # ======================== Ablation Switches ========================
        self.use_gated_fusion = getattr(self.conf, 'use_gated_fusion', 1)
        self.use_cl = getattr(self.conf, 'use_cl', 1)
        self.use_item_item = getattr(self.conf, 'use_item_item', 1)

        ########  Fusion Layer ########

        first_user_review_vector_matrix = self.convertDistribution(self.user_review_vector_matrix)
        first_item_review_vector_matrix = self.convertDistribution(self.item_review_vector_matrix)
        
        self.user_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_user_review_vector_matrix)
        self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_item_review_vector_matrix)

        second_user_review_vector_matrix = self.convertDistribution(self.user_reduce_dim_vector_matrix)
        second_item_review_vector_matrix = self.convertDistribution(self.item_reduce_dim_vector_matrix)

        self.fusion_item_embedding = self.item_embedding + second_item_review_vector_matrix
        self.fusion_user_embedding = self.user_embedding + second_user_review_vector_matrix

        ######## Influence and Interest Diffusion Layer ########

        # ----------------------
        # First Layer

        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems1(self.fusion_item_embedding)
        user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors1(self.fusion_user_embedding)

        consumed_items_attention = tf.math.exp(self.first_user_part_interest_graph_att_layer3(self.first_user_part_interest_graph_att_layer2(self.first_user_part_interest_graph_att_layer1(\
                                   tf.concat([self.fusion_user_embedding, user_embedding_from_consumed_items], 1))))) + 0.7
        social_neighbors_attention = tf.math.exp(self.first_user_part_social_graph_att_layer3(self.first_user_part_social_graph_att_layer2(self.first_user_part_social_graph_att_layer1(\
                                   tf.concat([self.fusion_user_embedding, user_embedding_from_social_neighbors], 1))))) + 0.3

        sum_attention = consumed_items_attention + social_neighbors_attention
        self.consumed_items_attention_1 = consumed_items_attention / sum_attention
        self.social_neighbors_attention_1 = social_neighbors_attention / sum_attention

        # MODIFIED: 第一层用户自适应残差融合 (可消融)
        first_user_neighbor_embedding = self.consumed_items_attention_1 * user_embedding_from_consumed_items \
                     + self.social_neighbors_attention_1 * user_embedding_from_social_neighbors
        if self.use_gated_fusion:
            first_user_residual_gate = self.first_user_residual_gate_layer(
                tf.concat([self.fusion_user_embedding, first_user_neighbor_embedding], 1)
            )
            first_gcn_user_embedding = first_user_residual_gate * self.fusion_user_embedding \
                 + (1.0 - first_user_residual_gate) * first_user_neighbor_embedding
        else:
            first_gcn_user_embedding = 0.5 * self.fusion_user_embedding + 0.5 * first_user_neighbor_embedding
        # MODIFIED END

        first_mean_social_influ, first_var_social_influ = tf.nn.moments(self.social_neighbors_attention_1, axes=0)
        first_mean_interest_influ, first_var_interest_influ = tf.nn.moments(self.consumed_items_attention_1, axes=0)
        self.first_layer_analy = [first_mean_social_influ, first_var_social_influ,\
                                  first_mean_interest_influ, first_var_interest_influ]

        item_itself_att = tf.math.exp(self.first_item_part_itself_graph_att_layer3(self.first_item_part_itself_graph_att_layer2(\
                          self.first_item_part_itself_graph_att_layer1(self.fusion_item_embedding)))) + 1.0

        item_customer_attenton = tf.math.exp(self.first_item_part_user_graph_att_layer3(self.first_item_part_user_graph_att_layer2(\
                          self.first_item_part_user_graph_att_layer1(self.generateItemEmebddingFromCustomer1(self.fusion_user_embedding))))) + 1.0

        item_sum_attention = item_itself_att + item_customer_attenton

        self.item_itself_att1 = item_itself_att / item_sum_attention
        self.item_customer_attenton1 = item_customer_attenton / item_sum_attention

        first_gcn_item_embedding = self.item_itself_att1 * self.fusion_item_embedding + \
            self.item_customer_attenton1 * self.generateItemEmebddingFromCustomer1(self.fusion_user_embedding)

        # MODIFIED: GraphTrust 物品-物品影响力 — 简单平均聚合 (可消融)
        if self.use_item_item:
            first_gcn_item_embedding = first_gcn_item_embedding + \
                self.generateItemEmebddingFromRelatedItems(self.fusion_item_embedding)

        first_mean_social_influ1, first_var_social_influ1 = tf.nn.moments(self.item_itself_att1, axes=0)
        first_mean_interest_influ1, first_var_interest_influ1 = tf.nn.moments(self.item_customer_attenton1, axes=0)
        self.first_layer_item_analy = [first_mean_social_influ1, first_var_social_influ1,\
                                  first_mean_interest_influ1, first_var_interest_influ1]

        # ----------------------
        # Second Layer

        user_embedding_from_consumed_items = self.generateUserEmebddingFromConsumedItems2(first_gcn_item_embedding)
        user_embedding_from_social_neighbors = self.generateUserEmbeddingFromSocialNeighbors2(first_gcn_user_embedding)

        consumed_items_attention = tf.math.exp(self.second_user_part_interest_graph_att_layer3(self.second_user_part_interest_graph_att_layer2(self.second_user_part_interest_graph_att_layer1(\
                                   tf.concat([first_gcn_user_embedding, user_embedding_from_consumed_items], 1))))) + 0.7

        social_neighbors_attention = tf.math.exp(self.second_user_part_social_graph_att_layer3(self.second_user_part_social_graph_att_layer2(self.second_user_part_social_graph_att_layer1(\
                                   tf.concat([first_gcn_user_embedding, user_embedding_from_social_neighbors], 1))))) + 0.3

        sum_attention = consumed_items_attention + social_neighbors_attention
        self.consumed_items_attention_2 = consumed_items_attention / sum_attention
        self.social_neighbors_attention_2 = social_neighbors_attention / sum_attention

        # MODIFIED: 第二层用户自适应残差融合 (可消融)
        second_user_neighbor_embedding = self.consumed_items_attention_2 * user_embedding_from_consumed_items \
                   + self.social_neighbors_attention_2 * user_embedding_from_social_neighbors
        if self.use_gated_fusion:
            second_user_residual_gate = self.second_user_residual_gate_layer(
                tf.concat([first_gcn_user_embedding, second_user_neighbor_embedding], 1)
            )
            second_gcn_user_embedding = second_user_residual_gate * first_gcn_user_embedding \
                 + (1.0 - second_user_residual_gate) * second_user_neighbor_embedding
        else:
            second_gcn_user_embedding = 0.5 * first_gcn_user_embedding + 0.5 * second_user_neighbor_embedding
        # MODIFIED END

        second_mean_social_influ, second_var_social_influ = tf.nn.moments(self.social_neighbors_attention_2, axes=0)
        second_mean_interest_influ, second_var_interest_influ = tf.nn.moments(self.consumed_items_attention_2, axes=0)
        self.second_layer_analy = [second_mean_social_influ, second_var_social_influ,\
                                   second_mean_interest_influ,second_var_interest_influ]

        item_itself_att = tf.math.exp(self.second_item_part_itself_graph_att_layer3(self.second_item_part_itself_graph_att_layer2(\
                          self.second_item_part_itself_graph_att_layer1(first_gcn_item_embedding)))) + 1.0

        item_customer_attenton = tf.math.exp(self.second_item_part_user_graph_att_layer3(self.second_item_part_user_graph_att_layer2(\
                          self.second_item_part_user_graph_att_layer1(self.generateItemEmebddingFromCustomer2(first_gcn_user_embedding))))) + 1.0

        item_sum_attention = item_itself_att + item_customer_attenton

        self.item_itself_att2 = item_itself_att / item_sum_attention
        self.item_customer_attenton2 = item_customer_attenton / item_sum_attention

        second_gcn_item_embedding = self.item_itself_att2 * first_gcn_item_embedding + \
            self.item_customer_attenton2 * self.generateItemEmebddingFromCustomer2(first_gcn_user_embedding)

        # MODIFIED: GraphTrust 第二层物品-物品影响力 — 简单平均聚合 (可消融)
        if self.use_item_item:
            second_gcn_item_embedding = second_gcn_item_embedding + \
                self.generateItemEmebddingFromRelatedItems(first_gcn_item_embedding)

        first_mean_social_influ2, first_var_social_influ2 = tf.nn.moments(self.item_itself_att2, axes=0)
        first_mean_interest_influ2, first_var_interest_influ2 = tf.nn.moments(self.item_customer_attenton2, axes=0)
        self.second_layer_item_analy = [first_mean_social_influ2, first_var_social_influ2,\
                                  first_mean_interest_influ2, first_var_interest_influ2]

        ######## Prediction Layer ########

        self.final_user_embedding = \
            tf.concat([first_gcn_user_embedding, second_gcn_user_embedding, self.user_embedding, second_user_review_vector_matrix], 1)
        self.final_item_embedding = \
            tf.concat([first_gcn_item_embedding, second_gcn_item_embedding, self.item_embedding, second_item_review_vector_matrix], 1)
        
        '''
        self.final_user_embedding = second_gcn_user_embedding
        self.final_item_embedding = second_gcn_item_embedding
        '''

        latest_user_latent = tf.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, self.item_input)
        
        self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))

        # ----------------------
        # Optimazation

        self.base_loss = tf.nn.l2_loss(self.labels_input - self.prediction)

        # ======================== 图结构对比学习模块 (可消融) ========================
        # 正样本: 图中真实相连的节点对 (社交边 / 物品共现边)
        # 负样本: 随机配对的节点对
        # 目标: 让图中相邻节点在嵌入空间中更接近

        self.use_item_cl = getattr(self.conf, 'use_item_cl', 0)

        if self.use_cl:
            self.cl_weight = getattr(self.conf, 'cl_weight', 0.01)
            # 用户侧: 社交边为正样本
            self.user_cl_loss = self.buildGraphContrastiveLoss(
                self.user_embedding,
                self.social_neighbors_indices_input,
                self.conf.num_users,
                tf.minimum(4096, tf.shape(self.social_neighbors_indices_input)[0])
            )
            # 物品侧: 共现边为正样本 (可独立关闭)
            if self.use_item_cl:
                self.item_cl_loss = self.buildGraphContrastiveLoss(
                    self.item_embedding,
                    self.item_item_indices_input,
                    self.conf.num_items,
                    tf.minimum(4096, tf.shape(self.item_item_indices_input)[0])
                )
            else:
                self.item_cl_loss = tf.constant(0.0)
        else:
            self.cl_weight = 0.0
            self.user_cl_loss = tf.constant(0.0)
            self.item_cl_loss = tf.constant(0.0)

        self.loss = self.base_loss
        self.opt_loss = self.base_loss \
            + self.cl_weight * (self.user_cl_loss + self.item_cl_loss)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v

        # MODIFIED: 保存自适应残差门控层参数
        for v in self.first_user_residual_gate_layer.variables:
            variables_dict[v.op.name] = v
        for v in self.second_user_residual_gate_layer.variables:
            variables_dict[v.op.name] = v
        # MODIFIED END

        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################

    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }
        
        map_dict['val'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST', 
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction,
            'train_base_loss': self.base_loss,
            'train_user_cl_loss': self.user_cl_loss,
            'train_item_cl_loss': self.item_cl_loss,
            'first_layer_ana': self.first_layer_analy,
            'second_layer_ana': self.second_layer_analy,
            'first_layer_item_ana': self.first_layer_item_analy,
            'second_layer_item_ana': self.second_layer_item_analy,
            'prediction': self.predict_vector,
            'user':self.final_user_embedding,
            'item':self.final_item_embedding,
            'low_att_user_user': self.first_user_user_low_att,
            'low_att_user_item': self.first_user_item_low_att,
            'low_att_user_user': self.first_item_user_low_att,
            'first_social_neighbors_low_att_matrix': self.first_social_neighbors_low_level_att_matrix,
            'second_social_neighbors_low_att_matrix': self.second_social_neighbors_low_level_att_matrix,
            'first_consumed_items_low_level_att_matrix':self.first_consumed_items_low_level_att_matrix,
            'second_consumed_items_low_level_att_matrix':self.second_consumed_items_low_level_att_matrix,
            'first_items_users_neighborslow_level_att_matrix':self.first_items_users_neighborslow_level_att_matrix,
            'second_items_users_neighborslow_level_att_matrix':self.second_items_users_neighborslow_level_att_matrix,
        }

        self.map_dict = map_dict
    

















