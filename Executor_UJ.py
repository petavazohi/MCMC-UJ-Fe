#!/usr/bin/env python
# Version 0.2

# Developers:
# -Aldo Romero               : alromero@mail.wvu.edu
# -Guillermo Avendano-Franco : gufranco@mail.wvu.edu
# -Pedram Tavadze            : petavazohi@mix.wvu.edu


import os
import argparse
import json
import time
import subprocess
import pychemia
import re

def analyze_output(code,params):
     if code == 'vasp':

          if os.path.isfile('results.json'):
               rf=open('results.json')
               full_json=json.load(rf)
               rf.close()
          else:
               full_json={}

          rf = open("OUTCAR","r")
          outcar_data = rf.read()
          rf.close()
          # To store results in JSON format
          jret={}

          U = params['U']
          J = params['J']
          name="U%.3f_J%.3f" % (U,J)


          if name in full_json:
               wf=open('single.json','w')
               json.dump(full_json[name], wf, sort_keys=True, indent=4, separators=(',', ': '))
               wf.close()
          
          else:
               structure = pychemia.code.vasp.read_poscar("POSCAR.inp")
               positions_before = structure.positions.tolist()

               structure = pychemia.code.vasp.read_poscar("CONTCAR")
               a = structure.lattice.a
               b = structure.lattice.b
               c = structure.lattice.c

               volume          = structure.volume
               natom           = structure.natom
               species         = structure.species
               positions_after = structure.positions.tolist()

               
               total_energy        = float(re.findall("energy\swithout\sentropy.*",outcar_data)[-1].split("=")[1].split()[0])
               inner_pressure_kbar = float(re.findall("external\spressure.*",outcar_data)[-1].split()[3])
               e_fermi             = float(re.findall("E-fermi.*",outcar_data)[-1].split()[2])
               magnetic_moments    = [map(float,x.split()[1:]) for x in re.findall("magnetization \(x\)[0-9a-z.\#\s\t\n-]*\ntot",outcar_data)[-1].split('\n')[4:4+natom]]

    
               # finding the band gap from EIGENVAL
               inputFile_EIGENVAL = open('EIGENVAL', 'r')
               # next f lines do not have anything interesting
               for i in range(5):
                    inputFile_EIGENVAL.readline()
               line = inputFile_EIGENVAL.readline()

               nelectrons     = int(line.split()[0])
               nkpt           = int(line.split()[1])
               neigen_per_kpt = int(line.split()[2])


               eigenup = []
               eigendown = []
               for i in range(nkpt):
                    eigenup.append([])
                    eigendown.append([])
                    inputFile_EIGENVAL.readline() # skips line before data
                    inputFile_EIGENVAL.readline() # this has kpoint and float weight
                    for j in range(neigen_per_kpt):
                         eigenvalue = map(float,inputFile_EIGENVAL.readline().split()[1:3])
                         eigenup[-1].append(eigenvalue[0])
                         eigendown[-1].append(eigenvalue[1])

               conduc_up   =  100.0
               conduc_down =  100.0
               valen_up    = -100.0
               valen_down  = -100.0

               for i in range(nkpt):
                    for eigenvalue in eigenup[i]:
                         if ((eigenvalue-e_fermi)<0.0):
                              valen_up    = max(valen_up,eigenvalue-e_fermi)
                         else:
                              conduc_up   = min(conduc_up,eigenvalue-e_fermi)
                    for eigenvalue in eigendown[i]:
                         if ((eigenvalue-e_fermi)<0.0):
                              valen_down  = max(valen_down,eigenvalue-e_fermi)
                         else:
                              conduc_down = min(conduc_down,eigenvalue-e_fermi)

               inputFile_EIGENVAL.close()
            
               jret['U']                          = U
               jret['J']                          = J
               jret['natom']                      = natom
               jret['species']                    = species
               jret['positions_before']           = positions_before
               jret['positions_after']            = positions_after
               jret['A_cell']                     = a
               jret['B_cell']                     = b
               jret['C_cell']                     = c
               jret['volume']                     = volume
               jret['total_energy']               = total_energy
               jret['inner_pressure_kbar']        = inner_pressure_kbar
               jret['e_fermi']                    = e_fermi
               jret['magnetic_moments']           = magnetic_moments
               jret['electrons']                  = nelectrons               
               jret['nkpt']                       = nkpt
               jret['neigenvalues_per_kpt']       = neigen_per_kpt               
               jret['valence_fermi_spin_down']    = valen_down
               jret['conduction_fermi_spin_down'] = conduc_down
               jret['gap_spin_down']              = conduc_down-valen_down
               jret['valence_fermi_spin_up']      = valen_up
               jret['conduction_fermi_spin_up']   = conduc_up
               jret['gap_spin_up']                = conduc_up-valen_up
               
               
               
               wf=open('single.json','w')
               json.dump(jret, wf, sort_keys=True, indent=4, separators=(',', ': '))
               wf.close()
               
               wf=open('results.json','w')
               full_json[name]=jret
               json.dump(full_json, wf, sort_keys=True, indent=4, separators=(',', ': '))
               wf.close()
     else : 
          raise ValueError("Not implemented for %s" % code)
     return full_json

def set_optimal_energy_cutoff(code, factor=1.4):
     
     if code == 'vasp':
          vi=pychemia.code.vasp.VaspInput('INCAR.inp')
          vi.set_encut(factor,POTCAR='POTCAR')
          vi.write('INCAR')
     else:
          raise ValueError("Not implemented for %s" % code)
     
def set_inputfile(code, params, calculation_type):
     if code == 'vasp':
          if calculation_type == "relaxation":
               # POSCAR Change
               st = pychemia.code.vasp.read_poscar('POSCAR.inp')
               pychemia.code.vasp.write_poscar(st,'POSCAR')

               # INCAR Change
               vi=pychemia.code.vasp.VaspInput('INCAR.inp')
               nspecies=st.nspecies
               arr=nspecies*[0]
               arr[0] = params['U']
               vi['LDAUU']=arr
               arr=nspecies*[0]
               arr[0] = params['J']
               vi['LDAUJ']  = arr
               vi['ISIF']   = 7
               vi['NSW']    = 100
	       vi['POTIM']  = 0.5
               vi['IBRION'] = 2

               vi.write('INCAR')

          elif calculation_type == "ground_state":
               # POSCAR Change
               st=pychemia.code.vasp.read_poscar('CONTCAR')
               pychemia.code.vasp.write_poscar(st,'POSCAR')

               # INCAR Change
               vi=pychemia.code.vasp.VaspInput('INCAR.inp')
               nspecies=st.nspecies
               arr=nspecies*[0]
               arr[0] = params['U']
               vi['LDAUU']=arr
               arr=nspecies*[0]
               arr[0] = params['J']
               vi['LDAUJ']=arr
               vi['ISYM'] = -1
               vi.write('INCAR')

     else:
          raise ValueError("Not implemented for %s" % code)

     return 

def execute(code, nparal):

     wf=open('RUNNING','w')
     wf.write(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
     wf.close()

     if code == 'vasp':
          start_time=time.time()
          status = subprocess.call("mpirun -np %d vasp_std "% nparal, shell=True)
          end_time=time.time()
          runtime=end_time-start_time
          if status== 0:
               print("VASP execution completed with returcode: %d runtime: %d secs" % (status, runtime))
          else:
               print("VASP execution failed with returcode: %d runtime: %d secs" % (status, runtime))
          os.remove('RUNNING')
     else:
          raise ValueError("Not implemented for %s" % code)

     return runtime



if __name__ == "__main__":
        
   description = ("Script_Run_JU.py: a python script to run VASP for a single or several J,U and with a possible scale change of volume (only works for single J,U).")
   parser = argparse.ArgumentParser(description=description)

   parser.add_argument('-np', dest='np', type=int, action='store', help='Number of MPI processes for the code')
   parser.add_argument('-code', dest='code', type=str, action='store', help='Code to use', default='vasp')
   parser.add_argument('-walltime', dest='walltime', type=int, action='store', help='Walltime in minutes')

   args=parser.parse_args()

   code = args.code
   walltime = args.walltime

   set_optimal_energy_cutoff(code, factor=1.4)

   jobstart=time.time()
   jobend=jobstart+walltime*60

   runtime=0


   if os.path.isfile('results.json'):
        rf=open('results.json')
        full_json=json.load(rf)
        rf.close()
   else:
        full_json={}

   while True:

        curtime=time.time()
        if runtime > jobend-curtime:
             print("Not enough time for one run")
             break
        
        if not os.path.isfile('input.json'):
             print("Not input.json found, waiting 60 seconds.  Time before wall: %d min" % int((jobend-time.time())/60))
             time.sleep(60)
             continue
        
        rf=open('input.json')
        data=json.load(rf)
        rf.close()
        
        print(data)

        if data['kind'] == 'singleUJ':
             U=float(data['U'])
             J=float(data['J'])
                          
             name="U%.3f_J%.3f" % (U, J)

             if not name in full_json:
                  set_inputfile(code, params={'U':U, 'J':J}, calculation_type = 'relaxation')
                  runtime = execute(code, args.np)
#                  set_inputfile(code, params={'U':U, 'J':J}, calculation_type = 'ground_state')
#                  runtime = execute(code, args.np)
             else:
                  print("Already calculated for U: %f J: %f " % (U,J))

             full_json=analyze_output(code,params={'U':U, 'J':J})


             wf=open('COMPLETE','w')
             wf.write(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
             wf.close()
             os.remove('input.json')
             

