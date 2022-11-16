import sys
import os
import glob
import argparse
import subprocess
from time import time

import yaml
import numpy as np
from ase.io import read, write
from pyxtal import pyxtal
from pyxtal.molecule import generate_molecules


def get_parser():
    class customHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                              argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=customHelpFormatter,
        description='Python script easy to make force field topology of molecular crystals',
    )
    parser.add_argument(
        '-i', '--inp', type=str,
        help = 'yaml style input file, overwriting argument values',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help = 'Verbose output.'
    )
    args = parser.parse_args()

    return args

def set_config(args):
    # Read config yaml file
    if args.inp is not None and os.path.isfile(args.inp):
        with open(args.inp, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}

    # Set up default config values from program arguments
    conf_def = vars(args).copy()
    [conf.setdefault(k, v) for k, v in conf_def.items()]

    return conf

def pyxtal_to_ase(cs, nxyz, mol_natoms, mol_atomtypes):
    nmols = cs.numMols[0] * nxyz[0] * nxyz[1] * nxyz[2]
    print('nxyz:', nxyz, 'nmols:', nmols)

    atoms = cs.to_ase(resort=False)
    atoms *= nxyz

    at = np.concatenate([mol_atomtypes[0] for i in range(nmols)])
    print(type(at), at)
    atoms.set_array('atomtypes', at)
    resname = 'M01'
    resname = np.array([resname for i in range(nmols*mol_natoms[0])])
    print(type(resname), resname)
    atoms.set_array('residuenames', resname)
    resnum = np.array([i // mol_natoms[0] + 1 for i in range(nmols*mol_natoms[0])])
    print(type(resnum), resnum)
    atoms.set_array('residuenumbers', resnum)

    return atoms

def get_uniq_chemical_symbols(atoms):
    return list(set(atoms.get_chemical_symbols()))

def run_cmd(cmd):
    print(' '.join(cmd))
    results = subprocess.run(cmd, capture_output=True, check=True, text=True)
    print('stdout:\n', results.stdout, '\nstderr:\n', results.stderr)
    return results

def g16resp_run(mol_path, g16inp_path, charge=0, multi=1, mem=4, nprocshared=4):
    atoms = read(mol_path, format='mol')

    s = '%mem={}GB\n%nprocshared={}\n'.format(mem, nprocshared)
    s += '#p hf/6-31g(d) pop=mk iop(6/33=2,6/42=6) scf=tight\n\n'
    s += 'HF/6-31g(d) RESP Charge\n\n'
    s += '{:d} {:d}\n'.format(charge, multi)
    for cs, xyz in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        s += '{:<2s} {:>15.8f} {:>15.8f} {:>15.8f}\n'\
            .format(cs, xyz[0], xyz[1], xyz[2])
    s += '\n'    
    print(s)

    with open(g16inp_path, mode='w') as f:
        f.write(s)

    cmd = ['g16', g16inp_path]
    run_cmd(cmd)

def psi4resp_run(mol_path, respchg_path='resp_charge.out'):
    import psi4
    import resp

    atoms = read(mol_path, format='mol')
    s = ''
    for cs, xyz in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        s += '{:<2s} {:>15.8f} {:>15.8f} {:>15.8f}\n'\
            .format(cs, xyz[0], xyz[1], xyz[2])
    #print(s)

    mol = psi4.geometry(s)
    mol.update_geometry()

    options = {'VDW_SCALE_FACTORS'  : [1.4, 1.6, 1.8, 2.0],
               'VDW_POINT_DENSITY'  : 1.0,
               'RESP_A'             : 0.0005,
               'RESP_B'             : 0.1,
           }

    # Call for RESP fit
    charges1 = resp.resp([mol], options)
    print('Electrostatic Potential Charges')
    print(charges1[0])
    print('Restrained Electrostatic Potential Charges')
    print(charges1[1])

    s = '\n'.join(['{:.6f}'.format(c) for c in charges1[1]])
    with open(respchg_path, mode='w') as f:
        f.write(s)

    return charges1[1]

def ac_run(pm, ff='gaff2', charge_model='bcc'):
    mol_atomtypes = []
    mol_natoms = []
    for i, p in enumerate(pm):
        resname = 'M{:02}'.format(i+1)
        molfile = resname + '.mol'
        mol2file = resname + '.mol2'
        pdbfile = resname + '.pdb'
        g16infile = resname + '.com'
        g16outfile = resname + '.log'
        respfile = resname + '_resp_charge.out'
        with open(molfile, mode='w') as f:
            f.write(p.rdkit_mb)

        if charge_model.upper() == 'RESP':
            g16resp_run(molfile, g16infile)
            cmd = ['antechamber', '-fi', 'gout', '-fo', 'mol2',  '-at', ff, '-c', 'resp', '-i', g16outfile, '-o', mol2file]
            #psi4resp_run(molfile, respfile)
            #cmd = ['antechamber', '-fi', 'mdl', '-fo', 'mol2', '-at', ff, '-c', 'rc', '-cf', respfile, '-rn', resname,'-i', molfile, '-o', mol2file]
        else:
            cmd = ['antechamber', '-fi', 'mdl', '-fo', 'mol2', '-at', ff, '-c', charge_model, '-rn', resname,'-i', molfile, '-o', mol2file]
        print(' '.join(cmd))
        run_cmd(cmd)

        cmd = ['antechamber', '-fi', 'mol2', '-fo', 'pdb', '-i', mol2file, '-o', pdbfile]
        print(' '.join(cmd))
        run_cmd(cmd)

        atoms = read(pdbfile, format='proteindatabank')
        at = atoms.get_array('atomtypes')
        mol_atomtypes.append(at)
        mol_natoms.append(len(at))

    return mol_atomtypes, mol_natoms

def tleap_run(pm, cryspdb_path='model.pdb', leapin_path='leap.in'):
    root, ext = os.path.splitext(cryspdb_path)
    prmtop_path = root + '.prmtop'
    crd_path = root + '.crd'

    s = 'source leaprc.gaff\n\n'
    for i, p in enumerate(pm):
        resname = 'M{:02}'.format(i+1)
        mol2_path = resname + '.mol2'
        s += '{} = loadmol2 {}\n'.format(resname, mol2_path)
        s += 'charge {}'.format(resname)
    s += '\n'
    s += 'crys = loadPDB {}\n'.format(cryspdb_path)
    s += 'charge crys\n\n'
    s += 'saveAmberParm crys {} {}\n'.format(prmtop_path, crd_path)
    s += 'quit\n'

    with open(leapin_path, mode='w') as f:
        f.write(s)

    cmd = ['tleap', '-f', leapin_path]
    print(' '.join(cmd))
    run_cmd(cmd)

    cell = []
    with open(cryspdb_path) as f:
        for l in f.readlines():
            if 'CRYST1' in l: 
                cell = l.split()[1:7]

    if len(cell) == 6:
        cmd = ['ChBox', '-c', crd_path, '-o', crd_path,
               '-X', cell[0], '-Y', cell[1], '-Z', cell[2],
               '-al', cell[3], '-bt', cell[4], '-gm', cell[5]]
        print(' '.join(cmd))
        run_cmd(cmd)

    return prmtop_path, crd_path

def parmed_run(amb_top_path, amb_crd_path):
    root, ext = os.path.splitext(amb_top_path)
    gmx_top_path = root + '.top'
    gmx_crd_path = root + '.gro'

    # convert topology from Amber to Gromacs
    import parmed as pmd
    amb_top = pmd.load_file(amb_top_path, xyz=amb_crd_path)
    amb_top.save(gmx_top_path, overwrite=True)
    amb_top.save(gmx_crd_path, overwrite=True)

    return gmx_top_path, gmx_crd_path

def intermol_run(amb_prmtop_path, amb_inpcrd_path):
    from intermol.convert import _load_amber, _save_lammps

    system, prefix, prmtop_in, crd_in, amb_structure \
        = _load_amber(amber_files=[amb_prmtop_path, amb_inpcrd_path])

    oname = '{0}'.format(prefix)
    output_status = dict()
    lmpin_path = '{0}.input'.format(oname)
    lmp_settings = {}
    lmp_settings['lmp_settings'] = 'pair_style lj/cut/coul/long 9.0 9.0\npair_modify tail yes\nkspace_style pppm 1e-8\n\n'
    _save_lammps(system, oname, output_status, lmp_settings)
    # Specify the output energies that we are interested in.
    energy_terms = " ".join(['ebond', 'eangle', 'edihed', 'eimp',
                             'epair', 'evdwl', 'ecoul', 'elong',
                             'etail', 'pe', 'pxx'])

    s = ''
    with open(lmpin_path) as f:
        for l in f.readlines():
            if 'thermo_style' in l:
                s += 'thermo_style custom {0}\n'.format(energy_terms)
            else:
                s += l

    with open(lmpin_path, mode='w') as f:
        f.write(s)

    return lmpin_path

def get_qepsdict(ps_path):
    elements = {}
    files = glob.glob(ps_path + "/*")
    for file in files:
        basename = os.path.basename(file)
        elem = basename[0:2].strip('_').strip('.').capitalize()
        elements[elem] = basename

    return elements

def write_proteindatabank(fileobj, images, write_arrays=True):
    """Write images to PDB-file."""

    if hasattr(images, 'get_positions'):
        images = [images]

    rotation = None
    if images[0].get_pbc().any():
        from ase.geometry import cell_to_cellpar, cellpar_to_cell

        currentcell = images[0].get_cell()
        cellpar = cell_to_cellpar(currentcell)
        exportedcell = cellpar_to_cell(cellpar)
        rotation = np.linalg.solve(currentcell, exportedcell)
        # ignoring Z-value, using P1 since we have all atoms defined explicitly
        format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
        fileobj.write(format % (cellpar[0], cellpar[1], cellpar[2],
                                cellpar[3], cellpar[4], cellpar[5]))

    #     1234567 123 6789012345678901   89   67   456789012345678901234567 890
    #format = ('ATOM  %5d %4s %3s     1    %8.3f%8.3f%8.3f%6.2f%6.2f'
    format = ('ATOM  %5d %4s %3s %4d    %8.3f%8.3f%8.3f%6.2f%6.2f'
              '          %2s  \n')

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)
    atomtypes = symbols
    if 'atomtypes' in images[0].arrays:
        atomtypes = images[0].get_array('atomtypes')

    for n, atoms in enumerate(images):
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        p = atoms.get_positions()
        if 'residuenames' in atoms.arrays:
            resnames = atoms.get_array('residuenames')
        else:
            resnames = np.array(['MOL' for i in range(len(atoms))])
        if 'residuenumbers'  in atoms.arrays:
            resnum = atoms.get_array('residuenumbers')
        else:
            resnum = np.ones(len(atoms), dtype=np.int8)
        occupancy = np.ones(len(atoms))
        bfactor = np.zeros(len(atoms))
        if write_arrays:
            if 'occupancy' in atoms.arrays:
                occupancy = atoms.get_array('occupancy')
            if 'bfactor' in atoms.arrays:
                bfactor = atoms.get_array('bfactor')
        if rotation is not None:
            p = p.dot(rotation)
        for a in range(natoms):
            x, y, z = p[a]
            occ = occupancy[a]
            bf = bfactor[a]
            fileobj.write(format % ((a+1) % MAXNUM, atomtypes[a], resnames[a], resnum[a],
                                    x, y, z, occ, bf, symbols[a].upper()))
        fileobj.write('ENDMDL\n')

def get_ase_calculator(atoms, sim_method='LAMMPS', top_path='model.input', kpts=(1,1,1), ps_path=None,
                       qe_input={'system': {'vdw_corr': 'DFT-D3', 'dftd3_version': 3}},
                       xtb_hamiltonian='GFN2-xTB'):
    if sim_method.upper() == 'LAMMPS':
        from molcrystop.lammpslib import LAMMPSlib
        calc = LAMMPSlib(lammps_input=top_path, log_file='lammps.log',
                         create_atoms=True, keep_alive=True)
    elif sim_method.upper() == 'AMBER':
        from ase.calculators.amber import Amber
        calc = Amber(amber_exe='sander -O ',
                     infile='mm.in',
                     outfile='mm.out',
                     topologyfile=top_path,
                     incoordfile='mm.crd')
    elif sim_method.upper() == 'DFTB':
        from ase.calculators.dftb import Dftb
        calc = Dftb(kpts=kpts,
                    Hamiltonian_='DFTB',
                    Hamiltonian_SCC='Yes',
                    Hamiltonian_SCCTolerance=1e-6,
                    Hamiltonian_MaxAngularMomentum_='',
                    Hamiltonian_MaxAngularMomentum_H='s',
                    Hamiltonian_MaxAngularMomentum_C='p',
                    Hamiltonian_MaxAngularMomentum_N='p',
                    Hamiltonian_MaxAngularMomentum_O='p',
                    Hamiltonian_MaxAngularMomentum_S='d',
                    Hamiltonian_MaxAngularMomentum_P='d',
                    Hamiltonian_MaxAngularMomentum_Br='d',
                    Hamiltonian_MaxAngularMomentum_Cl='d',
                    Hamiltonian_MaxAngularMomentum_F='p',
                    Hamiltonian_MaxAngularMomentum_I='d',
                    Hamiltonian_Dispersion_='DftD3',
                    Hamiltonian_Dispersion_Damping_='BeckeJohnson',
                    Hamiltonian_Dispersion_Damping_a1=0.746,
                    Hamiltonian_Dispersion_Damping_a2=4.191,
                    Hamiltonian_Dispersion_s6=1.0,
                    Hamiltonian_Dispersion_s8=3.209)
    elif sim_method.upper() == 'QE':
        from ase.calculators.espresso import Espresso
        pseudopotentials = get_qepsdict(ps_path)
        print('QE pseudopotentials:', pseudopotentials)
        input_data = qe_input
        print('QE input_data:', input_data)
        calc = Espresso(pseudopotentials=pseudopotentials,
                        tstress=True, tprnfor=True, kpts=kpts,
                        input_data=input_data)
    else:
        from ase.calculators.dftb import Dftb
        calc = Dftb(kpts=kpts,
                    Hamiltonian_='xTB',
                    Hamiltonian_Method=xtb_hamiltonian)

    return calc

def crys_geom_opt(ase_struc, sim_method='LAMMPS', top_path='model.input', kpts=(1,1,1), ps_path=None,
                  qe_input={'system': {'vdw_corr': 'DFT-D3', 'dftd3_version': 3}},
                  xtb_hamiltonian='GFN2-xTB',
                  opt_method='LBFGS', fmax=5e-2, opt_maxsteps=100, 
                  log_path='-', traj_path='crys_geom_opt.traj', maxstep=0.01, symprec=0.1):

    from ase.constraints import ExpCellFilter
    from ase.spacegroup.symmetrize import FixSymmetry

    calc = get_ase_calculator(
        atoms=ase_struc,
        sim_method=sim_method,
        top_path=top_path,
        kpts=kpts,
        ps_path=ps_path,
        qe_input=qe_input,
        xtb_hamiltonian=xtb_hamiltonian
    )
    ase_struc.calc = calc

    print('Initial lattice:')
    bl = ase_struc.cell.get_bravais_lattice()
    print(bl)
    e = ase_struc.get_potential_energy()
    print('Initial energy: {:.6f}'.format(e))
    f = ase_struc.get_forces()
    print('Initial force:\n', f)

    t1 = time()
    if opt_method == 'LBFGS':
        from ase.optimize import LBFGS
        opt = LBFGS(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
    elif opt_method == 'FIRE':
        from ase.optimize import FIRE
        opt = FIRE(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
    elif opt_method == 'LBFGSLineSearch':
        from ase.optimize import LBFGSLineSearch
        opt = LBFGSLineSearch(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
    elif opt_method == 'BFGS':
        from ase.optimize import BFGS
        opt = BFGS(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
    elif opt_method == 'GPMin':
        from ase.optimize import GPMin
        opt = GPMin(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path)
    elif opt_method == 'MDMin':
        from ase.optimize import MDMin
        opt = MDMin(ase_struc, logfile=log_path, trajectory=traj_path)
        opt.run(fmax=fmax, steps=opt_steps)
        from ase.optimize import LBFGS
        opt = LBFGS(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
    elif opt_method =='pyberny':
        from ase.optimize import Berny
        opt = Berny(ase_struc, logfile=log_path, trajectory=traj_path)
        opt.run(fmax=fmax, steps=opt_steps)
        from ase.optimize import LBFGS
        opt = LBFGS(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)
        #from ase.optimize import FIRE
        #opt = FIRE(ExpCellFilter(ase_struc), logfile=log_path, trajectory=traj_path, maxstep=maxstep)

    opt.run(fmax=fmax, steps=opt_maxsteps)
    t2 = time()
    t_opt = t2 - t1
    print('Geometry relaxation took {:3f} seconds.\n'.format(t_opt))

    opt_converged = opt.converged()
    print('Opt converged:', opt_converged)
    opt_steps = opt.nsteps #opt.get_number_of_steps()
    print('Number of optimized step', opt_steps)

    ase_struc.set_constraint()

    print('Final lattice:')
    bl = ase_struc.cell.get_bravais_lattice()
    print(bl)
    e = ase_struc.get_potential_energy()
    print('Final energy: {:.6f}'.format(e))
    f = ase_struc.get_forces()
    print('Final force:\n', f)

    basename = 'model_{}_opt'.format(sim_method)
    optgeom_path = basename + '.pdb'
    with open(optgeom_path, mode='w') as f:
        write_proteindatabank(f, ase_struc, write_arrays=True)
    optgeom_path = basename + '.cif'
    write(optgeom_path, ase_struc, format='cif')

    return opt_steps, opt_converged

def crystop_main(conf):

    inciffile = conf['input_file']
    smile = conf['smile']
    verbose = conf['verbose']
    ff = conf['ff']
    charge_model = conf['charge_model']

    qe_input = conf['qe_input']
    xtb_hamiltonian = conf['xtb_hamiltonian']

    nxyz = conf['nxyz']
    cryspdbfile = 'model.pdb'
    leapinfile = 'leap.in'

    geom_opt = conf['geom_opt']

    sim_methods = conf['sim_methods']
    kpts = conf['kpts']

    opt_method = conf['opt_method']
    fmax = conf['fmax']
    opt_maxsteps = conf['opt_maxsteps']

    dftb_command = conf['dftb_command']
    dftb_prefix = conf['dftb_prefix']

    espresso_command = conf['espresso_command']
    espresso_pseudo = conf['espresso_pseudo']

    #g16root = conf['g16root']
    #gauss_scrdir = conf['gauss_scrdir']

    ps_path = espresso_pseudo

    os.environ['ASE_DFTB_COMMAND'] = dftb_command
    os.environ['DFTB_PREFIX'] = dftb_prefix

    os.environ['ASE_ESPRESSO_COMMAND'] = espresso_command
    os.environ['ESPRESSO_PSEUDO'] = espresso_pseudo

    #os.environ['g16root'] = g16root
    #run_cmd(['source', g16root + '/bsd/g16.profile'])
    #os.environ['GAUSS_SCRDIR'] = gauss_scrdir
    #run_cmd(['which', 'g16'])

    pm = generate_molecules(smile, wps=None, N_iter=4, N_conf=10, tol=0.5)
    if verbose:
        for p in pm:
            print(vars(p))
            print(p.rdkit_mb)

    mol_atomtypes, mol_natoms = ac_run(pm, ff, charge_model)

    cs = pyxtal(molecular=True)
    cs.from_seed(inciffile, pm, add_H=True)
    if verbose:
        print(cs)
        print(vars(cs))

    # Create ASE atoms object
    atoms = pyxtal_to_ase(cs, nxyz, mol_natoms, mol_atomtypes)
    if verbose: print('atoms:', atoms)
    with open(cryspdbfile, mode='w') as f:
        write_proteindatabank(f, atoms, write_arrays=True)

    # Create Amber topology
    amb_prmtop_path, amb_inpcrd_path = tleap_run(pm, cryspdbfile, leapinfile)

    # Create Gromacs topology
    gmx_top_path, gmx_crd_path = parmed_run(amb_prmtop_path, amb_inpcrd_path)

    # Create Lammps topology
    lmpin_path = intermol_run(amb_prmtop_path, amb_inpcrd_path)
    print('lmpin_path:', lmpin_path)

    # Create DFTB+ coordinate
    #dftbin_path = 'model.gen'
    #write(dftbin_path, atoms, format='gen')

    if geom_opt:
        opt_stats = []
        for i, sim_method in enumerate(sim_methods):
            traj_path = 'model_{}_opt.traj'.format(sim_method)
            opt_steps, opt_coverged =  crys_geom_opt(
                ase_struc=atoms,
                sim_method=sim_method,
                top_path=lmpin_path,
                kpts=kpts,
                ps_path=ps_path,
                qe_input=qe_input,
                xtb_hamiltonian=xtb_hamiltonian,
                opt_method=opt_method,
                fmax=fmax[i],
                opt_maxsteps=opt_maxsteps[i],
                traj_path=traj_path
            )
            opt_stats.append([opt_steps, opt_coverged])

def main():
    args = get_parser()
    print(args)

    conf = set_config(args)

    print('======= Input configulations =======')
    for k, v in conf.items():
        print('{}: {}'.format(k, v))
    print('====================================')

    crystop_main(conf)

if __name__ == '__main__':
    main()
