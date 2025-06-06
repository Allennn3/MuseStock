from utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class Graph_Linear(nn.Module):
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x):
        output = torch.bmm(x.unsqueeze(1), self.W)
        output = output.squeeze(1)

        if self.bias:
            output = output + self.b
        return output


class Graph_Tensor(nn.Module):
    def __init__(self, num_stock, d_hidden, d_market, d_news, bias=True):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.d_news = d_news

        # Convolution layer for transforming news embeddings into hidden representations.
        self.seq_transformation_news = nn.Conv1d(d_news, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)

        # Define a learnable tensor graph parameter for modeling complex, multi-dimensional relationships between stocks.
        # Shape: (num_stock, d_hidden, d_hidden, d_hidden)
        self.tensorGraph = nn.Parameter(torch.zeros(num_stock, d_hidden, d_hidden, d_hidden))

        self.W = nn.Parameter(torch.zeros(num_stock, 2 * d_hidden, d_hidden))
        self.b = nn.Parameter(torch.zeros(num_stock, d_hidden))

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, market, news):

        t, num_stocks = news.size()[0], news.size()[1]

        # Reshape news tensor to flatten the time and stock dimensions for batch processing.
        news_transformed = news.reshape(-1, self.d_news)
        # Transpose the reshaped tensor to prepare for Conv1d layer input format (channels in second dimension).
        news_transformed = torch.transpose(news_transformed, 0, 1).unsqueeze(0)

        # Apply the news feature transformation using the 1D convolution layer to project into hidden space.
        news_transformed = self.seq_transformation_news(news_transformed)

        # Reshape and transpose the transformed news embeddings back to the original temporal structure for further processing.
        news_transformed = news_transformed.squeeze().transpose(0, 1)
        news_transformed = news_transformed.reshape(t, num_stocks, self.d_hidden)

        # Reshape and transpose market data to prepare for Conv1d input format: (batch_size=1, channels=d_market, length=t*num_stocks)
        market_transformed = market.reshape(-1, self.d_market)
        market_transformed = torch.transpose(market_transformed, 0, 1).unsqueeze(0)

        market_transformed = self.seq_transformation_markets(market_transformed)

        market_transformed = market_transformed.squeeze().transpose(0, 1)
        market_transformed = market_transformed.reshape(t, num_stocks, self.d_hidden)

        # Add two extra dimensions to prepare the news tensor for batched tensor multiplication with the learnable graph tensor.
        x_news_tensor = news_transformed.unsqueeze(2)
        x_news_tensor = x_news_tensor.unsqueeze(2)

        x_market_tensor = market_transformed.unsqueeze(-1)

        # Perform tensor multiplication with the learnable graph tensor to model high-order interactions between news and market data across stocks.
        temp_tensor = x_news_tensor.matmul(self.tensorGraph).squeeze()
        temp_tensor = temp_tensor.matmul(x_market_tensor).squeeze()

        # Concatenate transformed news and market embeddings along the feature dimension to form combined input for graph-based linear transformation.
        x_linear = torch.cat((news_transformed, market_transformed), axis=-1)

        temp_linear = torch.bmm(x_linear.transpose(0, 1), self.W)

        temp_linear = temp_linear.transpose(0, 1)

        # Apply hyperbolic tangent activation to the summed outputs from tensor interaction and linear transformation, along with bias.
        output = torch.tanh(temp_tensor + temp_linear + self.b)

        return output

class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):

        super(Graph_GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = Graph_Linear(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):

        super(Graph_GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)

        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
        return hidden

class Graph_Attention(nn.Module):

    def __init__(self, num_stock, in_features, out_features, dropout, alpha, concat=True, residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.alpha = alpha
        self.concat = concat

        self.residual = residual

        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.W_static = nn.Parameter(torch.zeros(num_stock, num_stock).type(torch.FloatTensor), requires_grad=True)

        self.w_1 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def sentiment_relation(self, input_r, input_s, sentiment_scores, lambda_factor=0.7):

        num_stock = input_r.shape[0]

        # Retrieve market signal-based relationships between stocks
        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)
        seq_s = torch.transpose(input_s, 0, 1).unsqueeze(0)

        # Initialize a logits matrix to store raw relationship scores between stocks
        logits = torch.zeros(num_stock, num_stock, device=input_r.device, dtype=input_r.dtype)

        # Transform the reshaped input_r using a 1D convolution layer to extract relation features
        seq_fts_r = self.seq_transformation_r(seq_r)

        # Apply two separate 1D convolutions to generate relation coefficients for graph construction
        f_1 = self.f_1(seq_fts_r)
        f_2 = self.f_2(seq_fts_r)

        # Update logits matrix by adding transformed relation features from f_1 and f_2
        logits += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)

        # Adjust relationship scores based on stock-specific features
        seq_fts_s = self.seq_transformation_s(seq_s)  # Process stock features for relation adjustment
        seq_fts_s = torch.transpose(seq_fts_s.squeeze(0), 0, 1)

        # Weight the relationship matrix using stock temporal feature similarity
        stock_temporal_feature_weight = torch.matmul(seq_fts_s, seq_fts_s.T)
        logits = logits * stock_temporal_feature_weight

        # sentiment factor
        pos_factor = sentiment_scores[:, 2]
        neg_factor = sentiment_scores[:, 0]

        sentiment_factor = pos_factor - neg_factor

        # Create a diagonal matrix from the sentiment factor vector
        sentiment_diag = torch.diag(sentiment_factor)

        graph = F.elu(logits)

        # Apply symmetric weighted multiplication using the sentiment diagonal matrix: S @ logits @ S
        sentiment_graph = torch.matmul(sentiment_diag, torch.matmul(graph, sentiment_diag)) * lambda_factor

        final_graph = sentiment_graph + graph

        # Remove self-connections to ensure only interactions between different stocks are considered
        if not isinstance(self.coef_revise, torch.Tensor):
            self.coef_revise = (
                    torch.ones(num_stock, num_stock, device=input_r.device) -
                    torch.eye(num_stock, device=input_r.device)
            )

        final_graph = final_graph * self.coef_revise

        return final_graph

    def get_gate(self, seq_s):
        transform_1 = self.w_1(seq_s)
        transform_2 = self.w_2(seq_s)

        transform_1 = torch.transpose(transform_1.squeeze(0), 0, 1)
        transform_2 = torch.transpose(transform_2.squeeze(0), 0, 1)

        gate = F.elu(transform_1.unsqueeze(1) + transform_2)
        return gate

    def forward(self, input_s, input_r, sentiment_scores):


        coefs_eye = self.sentiment_relation(input_r, input_s, sentiment_scores)


        seq_s = torch.transpose(input_s, 0, 1).unsqueeze(0)

        seq_fts_s = self.seq_transformation_s(seq_s)
        seq_fts_s = F.dropout(torch.transpose(seq_fts_s.squeeze(0), 0, 1), self.dropout, training=self.training)


        gate = self.get_gate(seq_s)


        seq_fts_s_gated = seq_fts_s * gate

        ret = torch.bmm(coefs_eye.unsqueeze(1), seq_fts_s_gated).squeeze()

        return torch.tanh(ret) if self.concat else ret


