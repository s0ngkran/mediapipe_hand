import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument('-tr','--train', help='train', action='store_true')
parser.add_argument('--test',  nargs=2, help='testtttttt\ntestttoteuth')
args = parser.parse_args()

print(args,'argsssss')
args.test[0]= 'hihihih'
print(args.test)
print(args.train)
