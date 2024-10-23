import argparse

class Parser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, help='gpu ids to use e.g. 0,1,2,...')
        self.parser.add_argument('--work-type', type=str, help='work-types i.e. gen_data or train')
        self.parser.add_argument('--model', type=str, help='model i.e. fedmatch')
        self.parser.add_argument('--task', type=str, default='lc-biid-c10', help='task i.e. lc-biid-c10, ls-bimb-c10')
        self.parser.add_argument('--frac-clients', type=float, help='fraction of clients per round')
        self.parser.add_argument('--seed', type=int, help='seed for experiment')
        self.parser.add_argument('--threshold', type=float, default=0.85, help='Threshold for confidence masking')
        self.parser.add_argument('--T', type=float, default=1.0, help='Temperature parameter for sharpening pseudo labels.')
        self.parser.add_argument('--dataset_id', type=int, required=True, help='Dataset ID for data loading')
        self.parser.add_argument('--num_test', type=int, default=2000, help='Number of test examples')  # SVHN 的测试集大小
        self.parser.add_argument('--num_valid', type=int, default=2000, help='Number of validation examples')
        self.parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
        self.parser.add_argument('--batch_size_test', type=int, default=100, help='batch_size_test')


    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
