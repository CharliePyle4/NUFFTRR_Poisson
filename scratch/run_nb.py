import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

nb_path = r"c:\Users\charl\NUFFTRR_Poisson\Tests\paper\paper_ex1.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
print("Executing paper_ex1.ipynb...")
ep.preprocess(nb, {'metadata': {'path': r"c:\Users\charl\NUFFTRR_Poisson"}})

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("paper_ex1.ipynb executed and saved successfully with outputs!")
