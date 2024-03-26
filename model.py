from torch import nn
import torch
import math

class NACCTemporal(nn.Module):
    def __init__(self, nhead=4, nlayers=3, hidden=128):
        super(NACCTemporal, self).__init__()

        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, enable_nested_tensor=False)

        # precompute position embeddings
        # we assume that no samples span more than 20 years
        self._posembds = self.__positionalencoding1d(hidden, 20)

    @staticmethod
    def __positionalencoding1d(d_model, length_max):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length_max, d_model)
        position = torch.arange(0, length_max).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, xs, temporal, temporal_mask):
        # copy down the relavent parts of the position embeddings
        embs = self._posembds.to(temporal.device)[temporal.int()] # (batches, pad_length, model_dim)

        # add it to the input
        net = xs + embs

        # create backplate
        shape = xs.shape
        backplate = torch.zeros((shape[0], shape[-1])).to(xs.device)
        
        # transformerify only the available part
        net = self.encoder(xs[~temporal_mask.all(dim=1)].transpose(0,1),
                           src_key_padding_mask=temporal_mask[~temporal_mask.all(dim=1)]).transpose(0,1)
        aggregate = net.mean(dim=1)
        backplate[~temporal_mask.all(dim=1)] = aggregate

        # average the sequence information
        return aggregate
        
class NACCFeatureExtraction(nn.Module):

    def __init__(self, nhead=4, nlayers=3, hidden=128):
        super(NACCFeatureExtraction, self).__init__()

        # the entry network ("linear value embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        self.__hidden = hidden

    def forward(self, x, mask):
        # don't forward pass on the padding; otherwise we'll nan
        is_padding = mask.all(dim=1)

        # so create all zero feature tensors and then insert non-padding
        # forward pass value in

        # the backplate to insert back
        backplate = torch.zeros((x.shape[0], self.__hidden)).float().to(x.device)

        # forward pass only on the non-paddng
        net = self.linear0(torch.unsqueeze(x[~is_padding], dim=2))
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask[~is_padding]).transpose(0,1)

        # put the results back
        backplate[~is_padding] = net.mean(dim=1)

        # average the output sequence information
        return backplate

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_classes, nhead=4, nlayers=3, hidden=128):
        # call early initializers
        super(NACCModel, self).__init__()

        # initial feature extraction system
        self.extraction = NACCFeatureExtraction(nhead, nlayers, hidden)
        self.temporal = NACCTemporal(nhead, nlayers, hidden)

        # prediction network
        self.ffnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
            nn.Softmax(1)
        )

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,
                feats_invariant, mask_invariant,
                feats_temporal, mask_temporal,
                timestamps,
                padding_mask, # True if its padding
                labels=None):

        # forward pass the input as one large batch
        stacked_inp = torch.cat([feats_invariant.unsqueeze(1),
                                   feats_temporal], dim=1)
        stacked_mask = torch.cat([mask_invariant.unsqueeze(1),
                                    mask_temporal], dim=1)
        # forward pass!
        os = stacked_inp.shape # save original shape, flatten the batch, forward, the reshape
        input_features = self.extraction(stacked_inp.reshape(os[0]*os[1],
                                                             os[2]),
                                         stacked_mask.reshape(os[0]*os[1],
                                                              os[2])).reshape(os[0], os[1], -1)

        # split out the temporal and non temporal layers
        inv_features = input_features[:, 0]
        temporal_features = input_features[:, 1:]

        # process the temporal features by another set of self attention
        temporal_features = self.temporal(temporal_features, timestamps, padding_mask)

        # fuse together and postprocess with a FFNN
        net = self.ffnn(temporal_features.nan_to_num(0) + inv_features)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

