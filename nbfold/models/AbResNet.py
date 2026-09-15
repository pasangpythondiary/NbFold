from os.path import isfile
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint

# from nbfold.models.PairedSeqLSTM import PairedSeqLSTM
from nbfold.resnets import ResNet1D, ResBlock1D, ResNet2D, ResBlock2D, RCCAModule
from nbfold.layers import OuterConcatenation2D
from nbfold.util.tensor import pad_data_to_same_shape


def create_output_block(out_planes2D, num_out_bins, kernel_size):
    return nn.Sequential(
        nn.Conv2d(out_planes2D,
                  num_out_bins,
                  kernel_size=kernel_size,
                  padding=kernel_size // 2),
        RCCAModule(in_channels=num_out_bins,
                   kernel_size=kernel_size,
                   return_attn=True))


class AbResNet(nn.Module):
    """
    Predicts binned output distributions for CA-distance, CB-distance, NO-distance, 
    omega and theta dihedrals, and phi planar angle from a one-hot encoded sequence 
    of heavy and light chain resides.
    """
    def __init__(self,
                 in_planes,                 
                 rnn_planes=128,
                 num_out_bins=37,
                 num_blocks1D=3,
                 num_blocks2D=25,
                 dilation_cycle=5,
                 dropout_proportion=0.2,
                 lstm_mean=None,
                 lstm_scale=None):
        super(AbResNet, self).__init__()
        
        # Define the linear layer
        self.linear = torch.nn.Linear(1024, 128)
        # Define the ReLU activation function
        self.relu = torch.nn.ReLU()

        self.output_names = [
            "ca_dist", "cb_dist", "no_dist", "omega", "theta", "phi"
        ]

        

        self._num_out_bins = num_out_bins
        self.resnet1D = ResNet1D(in_planes,
                                 ResBlock1D,
                                 num_blocks1D,
                                 planes=32,
                                 kernel_size=17)
        self.seq2pairwise = OuterConcatenation2D()

        # Calculate the number of planes output from the seq2pairwise layer
        out_planes1D = self.resnet1D.planes
        protT5=1024
        in_planes2D = 2 * (out_planes1D + protT5)

        self.resnet2D = ResNet2D(in_planes2D,
                                 ResBlock2D,
                                 num_blocks2D,
                                 planes=64,
                                 kernel_size=5,
                                 dilation_cycle=dilation_cycle)

        # Calculate the number of planes output from the ResNet2D layer
        out_planes2D = self.resnet2D.planes

        self.out_dropout = nn.Dropout2d(p=dropout_proportion)

        # Output convolution to reduce/expand to the number of bins
        self.out_ca_dist = create_output_block(out_planes2D, num_out_bins,
                                               self.resnet2D.kernel_size)
        self.out_cb_dist = create_output_block(out_planes2D, num_out_bins,
                                               self.resnet2D.kernel_size)
        self.out_no_dist = create_output_block(out_planes2D, num_out_bins,
                                               self.resnet2D.kernel_size)
        self.out_omega = create_output_block(out_planes2D, num_out_bins,
                                             self.resnet2D.kernel_size)
        self.out_theta = create_output_block(out_planes2D, num_out_bins,
                                             self.resnet2D.kernel_size)
        self.out_phi = create_output_block(out_planes2D, num_out_bins,
                                           self.resnet2D.kernel_size)

    

    def forward(self, x1,x2):
        # print("x1.shape one hot",x1.shape)
        # print("x2.shape transformer",x2.shape)
        out = self.resnet1D(x1)
        # print("renet1d.shape",out.shape)
        
        # lstm_enc = self.get_lstm_encoding(x)
        # x2=self.prot_lin(x2)
        # print(x2.shape)
        # x2=self.relu(self.linear(x2.transpose(1, 2)).transpose(1, 2))
        # print(x2.shape)
       
        out = torch.cat([out, x2], dim=1)
        # print("concat.shape",out.shape)

        out = self.seq2pairwise(out)
        # print("seq2pair.shape",out.shape)
        out = checkpoint(self.resnet2D, out)
        # print("checkpoint.shape",out.shape)
        out = self.out_dropout(out)
        # print("drop out shape",out.shape)

        out_ca_dist = self.out_ca_dist(out)[0]
        # print("out_ca_dist",out_ca_dist.shape)
        out_cb_dist = self.out_cb_dist(out)[0]
        # print("out_cb_dist",out_cb_dist.shape)
        out_no_dist = self.out_no_dist(out)[0]
        out_omega = self.out_omega(out)[0]
        out_theta = self.out_theta(out)[0]
        out_phi = self.out_phi(out)[0]

        out_ca_dist = out_ca_dist + out_ca_dist.transpose(2, 3)
        # print("out_ca_dist after transformation",out_ca_dist.shape)
        out_cb_dist = out_cb_dist + out_cb_dist.transpose(2, 3)
        # print("out_cb_dist after transformation",out_cb_dist.shape)
        out_omega = out_omega + out_omega.transpose(2, 3)
        

        return [
            out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta,
            out_phi
        ]

    def forward_attn(self, x1,x2):
        print("x1.shape one hot",x1.shape)
        print("x2.shape transformer",x2.shape)
        out = self.resnet1D(x1)
        print("renet1d.shape",out.shape)
        # lstm_enc = self.get_lstm_encoding(x)
        # x2=self.relu(self.linear(x2.transpose(1, 2)).transpose(1, 2))
        out = torch.cat([out, x2], dim=1)
        print("concat.shape",out.shape)

        out = self.seq2pairwise(out)
        print("seq2pair.shape",out.shape)
        out = checkpoint(self.resnet2D, out)
        print("checkpoint.shape",out.shape)
        out = self.out_dropout(out)
        print("drop out shape",out)

        out_ca_dist = self.out_ca_dist(out)[1]
        print("out_ca_dist",out_ca_dist)
        out_cb_dist = self.out_cb_dist(out)[1]
        print("out_cb_dist",out_cb_dist)
        out_no_dist = self.out_no_dist(out)[1]
        out_omega = self.out_omega(out)[1]
        out_theta = self.out_theta(out)[1]
        out_phi = self.out_phi(out)[1]
        

        return [
            out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta,
            out_phi
        ]


def load_model(model_file,
               eval_mode=True,
               device=None,
               scaled=True,
               strict=True):
    if not isfile(model_file):
        raise FileNotFoundError("No file at {}".format(model_file))
    checkpoint_dict = torch.load(model_file, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    dilation_cycle = 0 if not 'dilation_cycle' in checkpoint_dict else checkpoint_dict[
        'dilation_cycle']

    num_out_bins = checkpoint_dict['num_out_bins']
    in_planes = 21
    num_blocks1D = checkpoint_dict['num_blocks1D']
    num_blocks2D = checkpoint_dict['num_blocks2D']

    

    model = AbResNet(in_planes=in_planes,
                        num_out_bins=num_out_bins,
                        num_blocks1D=num_blocks1D,
                        num_blocks2D=num_blocks2D,
                        dilation_cycle=dilation_cycle)

    model.load_state_dict(model_state, strict=strict)

    # if device is not None:
    #     model = model.to(device)

    if eval_mode:
        model.eval()

    return model
