#!/usr/bin/env python

import os
import time
import pychemia

if __name__ == '__main__':

    user=os.getenv('USER')
    dirs=[x for x in os.listdir('.') if os.path.isdir(x)]
    while True:
        jobs = {}
        jobs_info=pychemia.runner.get_jobs(user)
        for job in jobs_info:
            name  = jobs_info[job]["Job_Name"]
            state = jobs_info[job]["job_state"]
            jobs[name] = state

        for idir in dirs:
            if idir not in jobs:
                job=pychemia.runner.PBSRunner(workdir=idir, template='template.pbs')
                if not os.path.exists(idir+os.sep+'Executor_UJ.py'):
                    os.symlink(os.path.abspath('Executor_UJ.py'), idir+os.sep+'Executor_UJ.py')
                job.set_pbs_params(nodes=1, ppn=16, queue='standby', mail="petavazohi.hpc@gmail.com",walltime=[4,0,0])
                job.submit()

        print("Sleeping for 60 seconds")
        time.sleep(60)
