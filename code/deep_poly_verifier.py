from abstract_shape import create_abstract_input_shape
from abstract_networks import AbstractNet1

def get_anet_class_from_name(net_name):
    abstract_nets = {
        'net1': AbstractNet1
    }
    return abstract_nets[net_name]


class DeepPolyVerifier:
    def __init__(self, net, net_name):
        abstract_net_class = get_anet_class_from_name(net_name)
        self.abstract_net = abstract_net_class(net)

    def verify(self, inputs, eps, true_label):
        abstract_input = create_abstract_input_shape(inputs, eps)

        curr_abstract_shape = abstract_input
        for abstract_transformer in self.abstract_net.get_abstract_transformers():
            curr_abstract_shape = abstract_transformer.forward(curr_abstract_shape)
        
        print(curr_abstract_shape)
        return True
