from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];TOOLS=ROOT/"tools/v3";sys.path.insert(0,str(TOOLS)) if str(TOOLS) not in sys.path else None
spec=importlib.util.spec_from_file_location("v9",TOOLS/"train_aee_corrective_composite_v9.py");assert spec and spec.loader
v9=importlib.util.module_from_spec(spec);spec.loader.exec_module(v9)
def test_fixed_role_mapping(): assert [v9.role_from_count(x) for x in (0,1,2,3,6)]==[2,2,0,1,1]
def test_macro_f1_exact():
 f,m=v9.macro_f1(np.array([1,2,3,4]),np.array([1,2,3,4]),(1,2,3,4));assert f==[1,1,1,1] and m==1
