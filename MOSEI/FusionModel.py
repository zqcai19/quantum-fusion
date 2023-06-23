import pennylane as qml
import torch
import torch.nn as nn
from math import pi
from Arguments import Arguments
args = Arguments()


def translator(net):
    assert type(net) == type([])
    updated_design = {}

    q = net[0:7]
    c = net[7:13]
    p = net[13:]

    # num of layer repetitions
    updated_design['layer_repe'] = 7

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        else:
            category = 'Ry'
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    pos_dict = {'00': 3, '01': 4, '10': 5, '11': 6}
    for j in range(args.n_qubits-1):
        if c[j] == 0:
            category = 'IsingXX'
        else:
            category = 'IsingZZ'
        if j <= 2:
            position = pos_dict[str(p[2*j]) + str(p[2*j+1])]
        else:
            position = j + 1
        updated_design['enta' + str(j)] = (category, [j, position])

    updated_design['total_gates'] = len(q) + len(c)
    return updated_design


qml.disable_return()
dev = qml.device("lightning.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(q_input_features_flat, q_weights_rot_flat, q_weights_enta_flat, **kwargs):
    current_design = kwargs['design']
    q_input_features = q_input_features_flat.reshape(args.n_qubits, 3)
    q_weights_rot = q_weights_rot_flat.reshape(current_design['layer_repe'], args.n_qubits)
    q_weights_enta = q_weights_enta_flat.reshape(current_design['layer_repe'], args.n_qubits-1)
    for layer in range(current_design['layer_repe']):
        # data reuploading
        for i in range(args.n_qubits):
            qml.Rot(*q_input_features[i], wires=i)
        # single-qubit parametric gates and entangled gates
        for j in range(args.n_qubits-1):
            if current_design['rot' + str(j)] == 'Rx':
                qml.RX(q_weights_rot[layer][j], wires=j)
            else:
                qml.RY(q_weights_rot[layer][j], wires=j)

            if current_design['enta' + str(j)][0] == 'IsingXX':
                qml.IsingXX(q_weights_enta[layer][j], wires=current_design['enta' + str(j)][1])
            else:
                qml.IsingZZ(q_weights_enta[layer][j], wires=current_design['enta' + str(j)][1])
        if current_design['rot' + str(args.n_qubits-1)] == 'Rx':
            qml.RX(q_weights_rot[layer][-1], wires=args.n_qubits-1)
        else:
            qml.RY(q_weights_rot[layer][-1], wires=args.n_qubits-1)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self, arguments, design):
        super(QuantumLayer, self).__init__()
        self.args = arguments
        self.design = design
        self.q_params_rot = nn.Parameter(pi * torch.rand(self.design['layer_repe'] * self.args.n_qubits))
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.design['layer_repe'] * (self.args.n_qubits-1)))

    def forward(self, input_features):
        q_out = torch.Tensor(0, self.args.n_qubits)
        q_out = q_out.to(self.args.device)
        for elem in input_features:
            q_out_elem = quantum_net(elem, self.q_params_rot, self.q_params_enta, design=self.design).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return q_out


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        # self.ClassicalLayer_a = nn.LSTM(self.args.a_insize, self.args.a_hidsize)
        # self.ClassicalLayer_v = nn.LSTM(self.args.v_insize, self.args.v_hidsize)
        # self.ClassicalLayer_t = nn.LSTM(self.args.t_insize, self.args.t_hidsize)
        self.ProjLayer_a = nn.Linear(self.args.a_insize, self.args.a_projsize)
        self.ProjLayer_v = nn.Linear(self.args.v_insize, self.args.v_projsize)
        self.ProjLayer_t = nn.Linear(self.args.t_insize, self.args.t_projsize)
        self.QuantumLayer = QuantumLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)

    def forward(self, x_a, x_v, x_t):
        # x = torch.permute(x, (1, 0, 2))
        # a_h = self.ClassicalLayer_a(x[:, :, :self.args.a_insize])[0][-1]
        # v_h = self.ClassicalLayer_v(
            # x[:, :, self.args.a_insize:self.args.a_insize+self.args.v_insize])[0][-1]
        # t_h = self.ClassicalLayer_t(
            # x[:, :, self.args.a_insize+self.args.v_insize:])[0][-1]
        a_p = torch.relu(self.ProjLayer_a(x_a))
        v_p = torch.sigmoid(self.ProjLayer_v(x_v)) * pi
        t_p = torch.sigmoid(self.ProjLayer_t(x_t)) * pi
        x_p = torch.cat((a_p, v_p, t_p), 1)
        exp_val = self.QuantumLayer(x_p)
        output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3
        return output
