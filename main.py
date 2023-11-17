import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import os.path as osp
import argparse
import random
import json


seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

print('Import completed')



def start(args):
    assert args.dataset in ['hugadb', 'pku-mmd'], 'Dataset not supported'

    device = torch.device(args.device)

    for i in range(1,2):
        print("Training subject: " + str(i))
        vid_list_file = osp.join(args.data_folder, args.dataset, 'splits_loso_validation', f'train.split{str(i)}.bundle')
        vid_list_file_tst = osp.join(args.data_folder, args.dataset, 'splits_loso_validation', f'test.split{str(i)}.bundle')   
        features_path = osp.join(args.data_folder, args.dataset, 'features7')  
        gt_path = osp.join(args.data_folder, args.dataset, 'groundTruth_')   

        print('Reading')
        mapping_file = osp.join(args.data_folder, args.dataset, 'mapping.txt') 
        dataset_params = json.load(open(osp.join(args.data_folder, args.dataset, 'dataset_parameters.json'), 'r'))
        print(dataset_params)

        

        model_dir = osp.join('.', 'models', args.dataset, f'split_{str(i)}')
        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        results_dir = osp.join('.', 'results', args.dataset, f'split_{str(i)}')
        if not osp.exists(results_dir):
            os.makedirs(results_dir)

        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])

        

        num_classes = len(actions_dict)
        print('Creating the trainer')
        trainer = Trainer(args.dil, args.num_layers_RF, args.num_stages, args.num_f_maps, dataset_params['C'], dataset_params['V'],
                           num_classes, args.graph_layout, args.graph_strategy)
        print(trainer.model.stream.graph.A.shape)
        if args.action == "train":
            batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
            batch_gen.read_data(vid_list_file)
            trainer.train(model_dir, batch_gen, num_epochs=args.num_epochs, batch_size=args.bz, learning_rate=args.lr, device=device)

        if args.action == "predict":
            trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, args.num_epochs, actions_dict, device, args.sample_rate)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    #Action
    parser.add_argument('--action', help='Action to perform, train or test', default='train')
    
    #Dataset
    parser.add_argument('--dataset', help='Dataset name', default="hugadb")
    parser.add_argument('--data_folder', help='Folder where the data are saved', default='.\\data')
    
    #Hyperparameters
    parser.add_argument('--num_stages', default=4)
    parser.add_argument('--num_layers_PG', default=10)
    parser.add_argument('--num_layers_RF', default=10)
    parser.add_argument('--num_f_maps', default=64)
    #parser.add_argument('--features_dim', default=6)
    parser.add_argument('--bz', default=16)
    parser.add_argument('--lr', default=0.0005)
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--dil', default=[1,2,4,8,16,32,64,128,256,512])
    parser.add_argument('--sample_rate', default=1)

    #Graph
    parser.add_argument('--graph_layout', default='hugadb')
    parser.add_argument('--graph_strategy', default='spatial')

    #Device
    parser.add_argument('--device', default='cpu')


    args = parser.parse_args()

    print(f'Start the {args.action}')
    start(args)
