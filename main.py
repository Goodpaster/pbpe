#!/usr/bin/env python3
from __future__ import print_function, division

def main(filename):
    '''Main Program.'''

    from src.read_input import read_input
    from src.pstr import pstr

    # initialize and print header
    pstr ("", delim="*", addline=False)
    pstr (" PROJECTION-BASED PERIODIC EMBEDDING ", delim="*", fill=False, addline=False)
    pstr ("", delim="*", addline=False)

    # print input file to stdout
    pstr ("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr ("End Input", addline=False)

    # get input
    inp = read_input(filename)

    # get supermolecular 1e matrices
    from src.integrals import get_supermol_1e_matrices
    inp = get_supermol_1e_matrices(inp)

    # get initial densities
    from src.integrals import get_subsystem_densities
    inp = get_subsystem_densities(inp)

    # periodic-in-periodic low level embedding
    from src.embedding import plow_in_plow
    inp = plow_in_plow(inp)

    # periodic high level-in-periodic low level embedding
    from src.embedding import phigh_in_plow
    inp = phigh_in_plow(inp)

    # finite cluster low level-in-periodic low level embedding
    from src.embedding import low_in_plow
    inp = low_in_plow(inp)

    # finite cluster high level-in-periodic low level embedding
    from src.embedding import high_in_plow
    inp = high_in_plow(inp)

#    # periodic CCSD calculation
#    from src.embedding import do_supermol_periodic_ccsd
#    do_supermol_periodic_ccsd(inp)

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

if __name__ == '__main__':

    filenames = get_input_files()
    for filename in filenames:
        obj = main(filename)
