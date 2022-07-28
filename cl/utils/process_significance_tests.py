import os
import numpy as np
import pandas as pd
import re

SIGNIFICANCE_TO_MAX_P = {
    "***": 0.001,
    "**": 0.01,
    "*": 0.05
}

def _hyp_path_to_mname(path):
    return path.split("/")[-3].replace("_ga", "").replace("noisy", "N").\
        replace("transfer_fixed", "trf").replace("transfer", "TR").\
        replace("adaptive_pacing", "AP").replace("seq_loss", "SEQ").\
        replace("_", "-").upper().replace("SUBSAMPLING", "SUB").\
        replace("SUBSAMPLE", "SUB").replace("BASELINE", "BASE").\
        replace("ASCENDING", "ASC").replace("-FULLASC", "")

def process_unified_file(filename, metric="MP"):
    with open(filename, 'r') as f:
        contents = [re.sub(
                "\s+", " ", 
                l.strip().replace("\t", " ")
            ) for l in f.readlines()]
        mnames = [re.findall("\|\s+Test\s+\|\|", l) for l in contents]
        mnames = contents[[i for i in range(len(mnames)) if len(mnames[i]) > 0][0]]
        mnames = re.sub("\|\s+Test\s+\|\|", "", mnames)
        mnames = [m.strip() for m in mnames.split("|") if len(m.strip())>0 and m.strip() != "Test"]
        mnames = list(map(_hyp_path_to_mname, mnames))
        contents = [l for l in contents if metric in l][1:]
        contents = [re.sub("\|\s+{}\s+\|".format(metric), "", l) for l in contents]
        contents = [[inner.strip() for inner in c.split("|") if len(inner.strip())>0] for c in contents]
        mnames_to_id = {m:i for i, m in enumerate(mnames)}
        arr = np.zeros((len(mnames),)*2)
        for c in contents:
            row = mnames_to_id[_hyp_path_to_mname(c[0])]
            for col in range(1, len(c)):
                sign_val = c[col].split(" ")[-1]
                sign_val = SIGNIFICANCE_TO_MAX_P.get(sign_val, sign_val)
                arr[row, row+col] = arr[row+col, row] = sign_val
        df = pd.DataFrame(arr, columns=mnames)
        df.index = mnames
        df[df>0.05] = "~"
        return df