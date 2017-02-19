import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v',"--verbosity", help="increase output verbosity",type=float)
args = parser.parse_args()
print args.verbosity
'''
if  args.verbosity:
    print "verbosity turned on " 
else:
	print 'bad'
'''