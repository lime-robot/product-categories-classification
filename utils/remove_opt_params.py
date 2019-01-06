import argparse
import torch
 
parser = argparse.ArgumentParser(description='Removes the optimizer parameter.')
parser.add_argument('--model', default='', type=str, metavar='PATH', required=True,
                    help='path to the model in which the optimizer parameter will be removed.')        
def main():
    args = parser.parse_args()    
    checkpoint = torch.load(args.model)
    checkpoint.pop('optimizer', None)
    torch.save(checkpoint, args.model)
    
if __name__ == '__main__':
    main()