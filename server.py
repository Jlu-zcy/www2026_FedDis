import torch
import torch.nn.functional as F
import numpy as np

class Server:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        self.similarity_method = 'cosine'
        self.temperature = self.args.temperature
        self.self_weight_boost = self.args.self_weight_boost
        

    def personalized_aggregate_params(self, client_params_list, client_embeddings, param_names_to_aggregate=None):
        if not client_params_list or not client_embeddings:
            return []              
        similarity_matrix = self._compute_similarity_matrix(client_embeddings)        
        attention_weights = self._compute_attention_weights(similarity_matrix)       
        personalized_params_list = self._aggregate_with_attention(
            client_params_list, attention_weights, param_names_to_aggregate
        )       
        personalized_params_list = self._process_traffic_pattern_parameters_separately(
            client_params_list, personalized_params_list
        )       
        return personalized_params_list

    def _compute_similarity_matrix(self, client_embeddings):       
        embeddings_matrix = torch.stack(client_embeddings)       
        normalized_embeddings = F.normalize(embeddings_matrix, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())        
        torch.diagonal(similarity_matrix).fill_(1.0)        
        return similarity_matrix

    def _compute_attention_weights(self, similarity_matrix):
        num_clients = similarity_matrix.shape[0]       
        scaled_similarities = similarity_matrix / self.temperature       
        if self.self_weight_boost > 1.0:
            diag_indices = torch.arange(num_clients)
            scaled_similarities[diag_indices, diag_indices] *= self.self_weight_boost       
        attention_weights = F.softmax(scaled_similarities, dim=1)      
        return attention_weights

    def _aggregate_with_attention(self, client_params_list, attention_weights, param_names_to_aggregate=None):
        num_clients = len(client_params_list)       
        if param_names_to_aggregate is None:
            param_names_to_aggregate = list(client_params_list[0].keys())       
        excluded_params = ['Bank', 'traffic_pattern_memory.W_p']
        param_names_to_aggregate = [name for name in param_names_to_aggregate if name not in excluded_params]       
        personalized_params_list = []      
        for i in range(num_clients):       
            client_weights = attention_weights[i]           
            personalized_params = {}            
            for param_name in client_params_list[i].keys():
                personalized_params[param_name] = client_params_list[i][param_name]           
            for param_name in param_names_to_aggregate:
                if param_name not in client_params_list[0]:
                    continue                
                param_list = []
                for j in range(num_clients):
                    if param_name in client_params_list[j]:
                        param_list.append(client_params_list[j][param_name])                
                if param_list:
                    param_tensor = torch.stack(param_list)                    
                    weight_shape = [num_clients] + [1] * (param_tensor.dim() - 1)
                    expanded_weights = client_weights.view(weight_shape)                    
                    personalized_param = torch.sum(expanded_weights * param_tensor, dim=0)
                    personalized_params[param_name] = personalized_param           
            personalized_params_list.append(personalized_params)                 
        return personalized_params_list

    def get_aggregation_config(self):
        return {
            'aggregate_all': False,
            'exclude_params': [],
            'include_params': [
                'st_encoder4invariant.encoder.dcrnn_cells.0.gate.weights_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.0.gate.bias_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.0.update.weights_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.0.update.bias_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.1.gate.weights_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.1.gate.bias_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.1.update.weights_pool', 
                'st_encoder4invariant.encoder.dcrnn_cells.1.update.bias_pool', 
                'st_encoder4invariant.end_conv.weight', 
                'st_encoder4invariant.end_conv.bias',
                'traffic_pattern_memory.W_p',
            ]
        }

    def _process_traffic_pattern_parameters_separately(self, client_params_list, personalized_params_list):
        num_clients = len(client_params_list)
        pattern_param_name = 'traffic_pattern_memory.W_p'
        
        pattern_params_list = []
        for i in range(num_clients):
            pattern_params_list.append(client_params_list[i][pattern_param_name])
        
        for i in range(num_clients):            
            current_patterns = pattern_params_list[i]
            N, c = current_patterns.shape            
            updated_patterns = current_patterns.clone()           
            for n in range(N):
                current_pattern = current_patterns[n]               
                similar_patterns = []
                pattern_weights = []
                total_contributing_clients = 0                
                for j in range(num_clients):
                    if i != j:
                        other_patterns = pattern_params_list[j]                        
                        similarities = []
                        for other_n in range(other_patterns.shape[0]):
                            other_pattern = other_patterns[other_n]
                            similarity = self._compute_feature_similarity(current_pattern, other_pattern)
                            similarities.append((other_n, similarity))                        
                        similarities.sort(key=lambda x: x[1], reverse=True)                       
                        similarity_threshold = getattr(self.args, 'pattern_similarity_threshold', 0.3)
                        max_patterns_per_client = getattr(self.args, 'pattern_top_k', 3)
                        
                        client_contributions = []
                        for other_n, sim in similarities:
                            if sim >= similarity_threshold:
                                client_contributions.append((other_patterns[other_n], sim))
                                if len(client_contributions) >= max_patterns_per_client:
                                    break
                        
                        if client_contributions:
                            total_contributing_clients += 1
                            for pattern, sim in client_contributions:
                                similar_patterns.append(pattern)
                                pattern_weights.append(sim)
                
                if similar_patterns:
                    similar_patterns_tensor = torch.stack(similar_patterns)
                    pattern_weights_tensor = torch.tensor(pattern_weights, device=similar_patterns_tensor.device)                    
                    normalized_weights = F.softmax(pattern_weights_tensor, dim=0)                    
                    aggregated_pattern = torch.sum(similar_patterns_tensor * normalized_weights.unsqueeze(1), dim=0)                    
                    current_weight = getattr(self.args, 'pattern_current_weight', 0.4)
                    updated_pattern = current_weight * current_pattern + (1 - current_weight) * aggregated_pattern                    
                    updated_patterns[n] = updated_pattern            
            personalized_params_list[i][pattern_param_name] = updated_patterns
        return personalized_params_list
    
    def _compute_feature_similarity(self, feature1, feature2):
        if not isinstance(feature1, torch.Tensor):
            feature1 = torch.tensor(feature1, dtype=torch.float32)
        if not isinstance(feature2, torch.Tensor):
            feature2 = torch.tensor(feature2, dtype=torch.float32)        
        similarity = F.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0), dim=1)
        
        return similarity.item()
    
    def _weighted_fusion_features(self, current_feature, similar_features, similarities):
        if not isinstance(current_feature, torch.Tensor):
            current_feature = torch.tensor(current_feature, dtype=torch.float32)       
        similarities_tensor = torch.tensor(similarities, dtype=torch.float32)
        weights = F.softmax(similarities_tensor, dim=0)     
        current_weight = getattr(self.args, 'bank_current_weight', 0.4)       
        fused_feature = current_weight * current_feature       
        for i, similar_feature in enumerate(similar_features):
            if not isinstance(similar_feature, torch.Tensor):
                similar_feature = torch.tensor(similar_feature, dtype=torch.float32)           
            fused_feature += (1 - current_weight) * weights[i] * similar_feature
        
        return fused_feature
