#!/usr/bin/env python
from __future__ import print_function, division

def main(filename):
    '''Main Program.'''

    from read_input import read_input

    # initialize and print header
    pstr ("", delim="*", addline=False)
    pstr (" EMBEDDED PERIODIC BOUNDARY CONDITIONS ", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)

    # print input file to stdout
    pstr ("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr ("End Input", addline=False)

    # get input
    inp = read_input(filename)

    # get initial densities
    from integrals import get_subsystem_densities
    inp = get_subsystem_densities(inp)

    # get supermolecular 1e matrices
    from integrals import get_supermol_1e_matrices
    inp = get_supermol_1e_matrices(inp)

    # periodic-in-periodic low level embedding
    from embedding import plow_in_plow
    inp = plow_in_plow(inp)

    # periodic high level-in-periodic low level embedding
    from embedding import phigh_in_plow
    inp = phigh_in_plow(inp)

    # finite cluster low level-in-periodic low level embedding
    from embedding import low_in_plow
    inp = low_in_plow(inp)

    # finite cluster high level-in-periodic low level embedding
    from embedding import high_in_plow
    inp = high_in_plow(inp)

    # close timer and h5py files
    inp.h5py.close()
    inp.timer.close()

    # return
    return inp

def get_input_files():
    '''Get input filenames from command line.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    # parse input files
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('input_files', nargs='*', default=sys.stdin,
                        help='The input files to submit.')
    args = parser.parse_args()

    return args.input_files

def pstr(st, delim="=", l=80, fill=True, addline=True, after=False):
    '''Print formatted string <st> to output'''
    if addline: print ("")
    if len(st) == 0:
        print (delim*l)
    elif len(st) >= l:
        print (st)
    else:
        l1 = int((l-len(st)-2)/2)
        l2 = int((l-len(st)-2)/2 + (l-len(st)-2)%2)
        if fill:
            print (delim*l1+" "+st+" "+delim*l2)
        else:
            print (delim+" "*l1+st+" "*l2+delim)
    if after: print ("")

if __name__ == '__main__':

    filenames = get_input_files()
    for filename in filenames:
        obj = main(filename)
