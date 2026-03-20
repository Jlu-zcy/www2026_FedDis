import copy
import torch
from train import Trainer
from models.graph_attention import GraphAttention

class Client:
    def __init__(self, cid, model, dataloader, args, logger, graph, aggregation_method='mean'):
        self.cid = cid
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.args = args
        self.logger = logger
        self.graph = graph
        self.current_lr = self.args.lr_init
        
        self.aggregation_method = self.args.aggregation_method
        
        if self.aggregation_method == 'attention':
            self.graph_aggregator = GraphAttention(
                input_dim=args.embed_dim,
                hidden_dim=args.d_model,
                output_dim=args.d_model,
                dropout=0.1
            )
        
        self._move_aggregator_to_device()
        self.num_nodes = self._get_num_nodes()        

    def _get_num_nodes(self):        
        if hasattr(self.model, 'globalST_encoder') and hasattr(self.model.globalST_encoder, 'node_embeddings'):
            return self.model.globalST_encoder.node_embeddings.shape[0]
        elif hasattr(self.model, 'personalST_encoder') and hasattr(self.model.personalST_encoder, 'node_embeddings'):
            return self.model.personalST_encoder.node_embeddings.shape[0]
        elif hasattr(self.args, 'num_nodes'):
            return self.args.num_nodes
            
    def local_train(self, local_epochs):
        if hasattr(self, '_cached_embeddings'):
            self._cached_embeddings = None
            
        return self.train_with_attention(local_epochs)

    def train_with_attention(self, local_epochs):
        all_params = self.get_optimizer_parameters()
        optimizer = torch.optim.Adam(all_params, lr=self.current_lr)
        
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            dataloader=self.dataloader,
            graph=self.graph,
            graph2=self.graph,
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=self.args.lr_patience, 
                threshold=0.0001, threshold_mode='rel', min_lr=0.000005, eps=1e-08),
            args=self.args,
            logger=self.logger,
            client=self
        )
        
        from federated import local_train_with_attention
        self.model, train_info = local_train_with_attention(
            trainer, local_epochs, self.args, self.cid
        )
        
        self.last_train_info = train_info
        return self.model

    def get_model_params(self):
        return self.model.state_dict()

    def get_partial_params(self, param_names):
        full_params = self.model.state_dict()
        partial_params = {}
        for name in param_names:
            if name in full_params:
                partial_params[name] = full_params[name]
        return partial_params

    def get_graph_embedding(self, input_data=None):
        self.model.eval()
        with torch.no_grad():
            if input_data is None:
                for batch in self.dataloader:
                    input_data = batch
                    break               
            node_embeddings = self._extract_node_embeddings()                               
            graph_embedding, aggregation_info = self.graph_aggregator(node_embeddings)                                
            return graph_embedding, aggregation_info

    def _extract_node_embeddings(self, use_cache=True):
        if use_cache and hasattr(self, '_cached_embeddings') and self._cached_embeddings is not None:
            return self._cached_embeddings
        if hasattr(self.model, 'globalST_encoder') and hasattr(self.model.globalST_encoder, 'node_embeddings'):
            node_embeddings = self.model.globalST_encoder.node_embeddings
            if use_cache:
                self._cached_embeddings = node_embeddings.detach()
            return node_embeddings
        elif hasattr(self.model, 'personalST_encoder') and hasattr(self.model.personalST_encoder, 'node_embeddings'):
            node_embeddings = self.model.personalST_encoder.node_embeddings
            if use_cache:
                self._cached_embeddings = node_embeddings.detach()
            return node_embeddings
            
    def update_model_params(self, aggregated_params, update_strategy='replace'):
        current_params = self.model.state_dict()
        
        if update_strategy == 'replace':
            current_params.update(aggregated_params)
        elif update_strategy == 'partial':
            for name, param in aggregated_params.items():
                if name in current_params:
                    current_params[name] = param
        
        self.model.load_state_dict(current_params)

    def set_model_params(self, state_dict):
        self.model.load_state_dict(state_dict) 

    def _move_aggregator_to_device(self):
        model_device = next(self.model.parameters()).device           
        self.graph_aggregator = self.graph_aggregator.to(model_device)
            
    def get_optimizer_parameters(self):
        return list(self.model.parameters()) + list(self.graph_aggregator.parameters())
