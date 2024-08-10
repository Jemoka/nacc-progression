from torch import nn
import torch
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NACCTemporalLSTM(nn.Module):
    def __init__(self, nlayers=3, hidden=128):
        super().__init__()

        # the encoder network
        self.lstm = nn.LSTM(hidden, hidden, nlayers)

        # precompute position embeddings
        # we assume that no samples span more than 50 years
        self._posembds = self.__positionalencoding1d(hidden, 50)
        self._nlayers = nlayers

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

    def forward(self, xs, data_mask,
                timestamps, prediction_timestamp,
                temporal_mask):
        res = torch.zeros((xs.size(0), xs.size(-1)), device=xs.device)

        # compute position embeds for final prediction
        out_temporal_embs = self._posembds.to(timestamps.device)[prediction_timestamp.int()]

        # compute sequence lengths of input
        seq_lens = (~temporal_mask).sum(1)

        # if we have no temporal data, we skip all the LSTM
        if (sum(seq_lens) == 0).all():
            return res # + out_temporal_embs

        # packem!
        packed = pack_padded_sequence(xs[seq_lens > 0], seq_lens[seq_lens > 0].cpu().tolist(),
                                      batch_first=True, enforce_sorted=False)
        # brrrr
        _, (out, __) = self.lstm(packed)
        # squash down hidden dims by averaging it
        out = out.sum(dim=0)
        # create a backplate for non-temporal data (i.e. those with seq_lens < 0)
        # insert these outputs from above as well as raw x for non temporal data
        res[seq_lens > 0] = out

        # and return everything we got
        return res
        
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
class NACCFuseModel(nn.Module):

    def __init__(self, num_classes, num_features, nhead=4, nlayers=3, hidden=128):
        # call early initializers
        super().__init__()

        # initial feature extraction system
        self.extraction = NACCFeatureExtraction(nhead, nlayers, hidden)
        self.temporal = NACCTemporalLSTM(nlayers, num_features)
        self.hidden = hidden

        # create a mapping between feature and hidden space
        # so temporal can be fused with hidden
        # we don't have bias to ensure zeros stay zeros
        self.proj = nn.Linear(num_features, hidden, bias=False)

        # mix attention projection
        self.offset = nn.Parameter(torch.rand(1), requires_grad=True)

        self.Q_proj = nn.Linear(hidden, hidden, bias=False)
        self.K_proj = nn.Linear(hidden, hidden, bias=False)
        self.V_proj = nn.Linear(hidden, hidden, bias=False)

        # prediction network
        self.ffnn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
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
                prediction_timestamp, # the timestamps which the labels are at
                labels=None):
        # encnode the invariant featrues first
        invariant_encoding = self.extraction(feats_invariant, mask_invariant)
        # and encode the temporal features
        temporal_encoding = self.temporal(feats_temporal, mask_temporal,
                                          timestamps, prediction_timestamp,
                                          padding_mask)

        # late fuse and predict
        # apply a learned offset shift to the temporal data
        # we do this instead of bias to ensure that each slot
        # recieves the same offset value if no temporal
        temporal_encoding = self.proj(temporal_encoding)
        offset_encoding = temporal_encoding + self.offset[0]
        paired_seq = torch.stack([invariant_encoding, offset_encoding], dim=-2)

        Q = self.Q_proj(paired_seq)
        K = self.K_proj(paired_seq)
        V = self.V_proj(paired_seq)

        attn_scores = (torch.einsum("blh,bsh -> bls", Q, K)/
                       (self.hidden**0.5)).softmax(dim=-1)
        fused = (attn_scores @ V).sum(dim=-2)

        net = self.ffnn(fused)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

