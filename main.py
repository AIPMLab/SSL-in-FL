import os
from custom_parser import Parser
from datetime import datetime
from data.loader import DataLoader
from misc.utils import *
from config import *

def main(args):
    print(f"Received arguments: {args}")
    args = set_config(args)

    print(f"Setting CUDA_VISIBLE_DEVICES to {args.gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    now = datetime.now().strftime("%Y%m%d-%H%M")
    args.log_dir = f'{args.output_path}/logs/{now}-{args.model}-{args.task}'
    args.check_pts = f'{args.output_path}/check_pts/{now}-{args.model}-{args.task}'

    print(f"Log directory set to {args.log_dir}")
    print(f"Check points directory set to {args.check_pts}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.check_pts, exist_ok=True)

    if args.work_type == 'gen_data':
        from data.generator import DataGenerator
        dgen = DataGenerator(args)
        dgen.generate_data()
        print(f"Data generation completed for task: {args.task}")

    elif args.model == 'fedmatch':
        print(f"Selected model: {args.model}")
        from models.fedmatch.server import Server
        server = Server(args)
        print("Running server...")
        server.run()
    else:
        print('Incorrect model was given: {}'.format(args.model))
        os._exit(0)

if __name__ == '__main__':
    main(Parser().parse())
