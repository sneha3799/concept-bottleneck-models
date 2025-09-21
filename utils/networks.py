import os
import torch
from torch import nn
from torchvision import models
from utils.model_utils import unfreeze_module, freeze_module

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class CBMNetwork(nn.Module):

    def __init__(self, 
                args):

        super(CBMNetwork, self).__init__()

        self.args = args

        self.num_concepts = args.num_concepts
        self.num_classes = args.num_classes
        self.encoder_arch = args.encoder_arch
        self.head_arch = args.head_arch
        self.training_mode = args.training_mode
        self.concept_learning = args.concept_learning

        self.setup(args)

    def setup(self, args):

        if self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        args.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")

        self.concept_predictor = nn.Linear(n_features, self.num_concepts, bias=True)
        self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.concept_dim, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(self, x):

        intermediate = self.encoder(x)
        c_logit = self.concept_predictor(intermediate)
        c_prob = self.act_c(c_logit)
        y_pred_logits = self.head(c_logit)

        return c_prob, y_pred_logits

    def intervene(self, concepts_interv_probs):

        c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
        y_pred_logits = self.head(c_logit)
        return y_pred_logits
    
    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)