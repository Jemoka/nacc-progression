from torch import nn
import torch

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, nhead=4, nlayers=3, hidden=128):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, enable_nested_tensor=False)

        # flatten!
        self.flatten = nn.Flatten()

        # dropoutp!
        self.dropout = nn.Dropout(0.5)

        # encoding network
        self.emb_enc = nn.Sequential(
            nn.Linear(hidden*num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # comparison layer
        self.cmp = nn.Linear(hidden, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def encodify(self, x, mask, detach_transformer):
        net = self.linear0(torch.unsqueeze(x, dim=2))
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)

        # if needed to freeze transformer weights
        if detach_transformer:
            net = net.detach()

        net = self.flatten(net)
        net = self.relu(net)
        net = self.dropout(net)
        return self.emb_enc(net)

    def forward(self, x, mask, labels=None, detach_transformer=False):

        # pass left and right through network
        l,r = x[:, 0, :], x[:, 1, :]
        lm,rm = mask[:, 0, :], mask[:, 1, :]

        # encode the sum of both sides
        latent_left = self.encodify(l, lm, detach_transformer)
        latent_right = self.encodify(r, rm, detach_transformer)

        combined = latent_left + latent_right

        # ffnn for comparison
        comparison = self.softmax(self.cmp(combined))

        # calculate loss, if needed
        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(comparison, labels)

        return { "logits": comparison, "loss": loss, "latents": [latent_left.detach(), latent_right.detach()] }

