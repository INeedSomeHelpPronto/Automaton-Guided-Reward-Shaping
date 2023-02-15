import os
from pathlib import Path
import subprocess



base = 'cass_exp/sep7/no_agrs/'
entries = Path('cass_exp/sep7/no_agrs/')
out_dir = 'cass_exp/sep7/van_out'


i = 0


for entry in entries.iterdir():
    # then it is an event

    with os.scandir(base + entry.name) as inner:
        for inn in inner:

            if inn.name[0] == 'e':

                stream = subprocess.Popen(["python", "evaluation/exportTensorFlowLog.py", base + entry.name, out_dir, "scalars,tr" + str(i)],
                                        stdin =subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True,
                                        bufsize=0)
                i+=1

        # Fetch output
        for line in stream.stdout:
            print(line.strip())


            ##### in order to merge all of these logs together for analysis...use merge-csv.com