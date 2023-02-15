import os
from pathlib import Path
import subprocess



base = 'cass_exp/new_pqr/pre/agrs/'
entries = Path('cass_exp/new_pqr/pre/agrs/')
out_dir = 'cass_exp/new_pqr/pre/agrs'


i = 0

GET_LOGS_FOR = 'pit'

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