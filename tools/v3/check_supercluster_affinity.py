"""Execute unchanged upstream definitions with demand-loaded dependencies.

This bypasses upstream eager package __init__ imports, NOT its algorithms.
No substitute CSR, majority, graph trimming or scatter implementations.
Not a full official entrypoint/model reproduction.
"""
import ast,builtins,hashlib,importlib,json,subprocess,types
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]/'build/gse_supercluster_reference_v1'
COMMIT='eb959b61226f60e0c037cc105c56d51318650e8a'
SPACES={};READS={};EXECUTED=[]

def namespace(module):
    if module not in SPACES:SPACES[module]=SourceNamespace(module)
    return SPACES[module]

def resolve(module,name):
    if module=='src.utils':
        hits=[]
        for p in (ROOT/'src/utils').glob('*.py'):
            tree=ast.parse(p.read_text())
            if any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name for n in tree.body):hits.append('src.utils.'+p.stem)
        if len(hits)!=1:raise ImportError(f'nonunique upstream export {name}: {hits}')
        module=hits[0]
    return namespace(module)[name]

class SourceNamespace(dict):
    def __init__(self,module):
        super().__init__(__builtins__=builtins.__dict__,__name__=module)
        self.module=module;self.path=ROOT/(module.replace('.','/')+'.py')
        b=self.path.read_bytes();READS[str(self.path.relative_to(ROOT))]=hashlib.sha256(b).hexdigest()
        self.tree=ast.parse(b,filename=str(self.path));self.nodes={};self.imports={}
        for n in self.tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):self.nodes[n.name]=n
            elif isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name):self.nodes[target.id]=n
            elif isinstance(n,ast.Import):
                for a in n.names:self.imports[a.asname or a.name.split('.')[0]]=(a.name,None)
            elif isinstance(n,ast.ImportFrom):
                for a in n.names:self.imports[a.asname or a.name]=(n.module,a.name)
        # Class-body annotations use globals directly rather than __missing__.
        for key,(module,name) in self.imports.items():
            if not module.startswith('src'):
                value=importlib.import_module(module)
                self[key]=getattr(value,name) if name is not None else value
    def __missing__(self,key):
        if key in builtins.__dict__:return builtins.__dict__[key]
        if key in self.nodes:
            node=self.nodes[key]
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(self.path),'exec'),self)
            EXECUTED.append(self.module+':'+key)
            return dict.__getitem__(self,key)
        if key in self.imports:
            module,name=self.imports[key]
            if module=='src' and name is None:
                value=types.SimpleNamespace(is_debug_enabled=resolve('src.debug','is_debug_enabled'))
            elif module.startswith('src.'):
                if name is None:raise ImportError('unsupported module import '+module)
                value=resolve(module,name)
            else:
                value=importlib.import_module(module)
                if name is not None:value=getattr(value,name)
            self[key]=value;return value
        raise KeyError(key)

def main():
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    cls=resolve('src.data.instance','InstanceData')
    # Three clusters: pure object0, 75% object0+25% object1, pure object1.
    obj=cls(torch.tensor([0,1,3,4]),torch.tensor([0,0,1,1]),torch.tensor([4,3,1,4]),torch.tensor([0,0,0,0]))
    edges=torch.tensor([[0,1,1,2,0],[1,0,2,1,0]])
    e,a=obj.instance_graph(edges,num_classes=1,smooth_affinity=True)
    torch.testing.assert_close(e,torch.tensor([[0,1],[1,2]]))
    torch.testing.assert_close(a,torch.tensor([.875,.125]))
    _,hard=obj.instance_graph(edges,num_classes=1,smooth_affinity=False)
    torch.testing.assert_close(hard,torch.tensor([1.,0.]))
    empty_e,empty_a=obj.instance_graph(torch.empty(2,0,dtype=torch.long),num_classes=1)
    assert empty_e.shape==(2,0) and empty_a.numel()==0
    # Reversing edges cannot change undirected targets.
    re,ra=obj.instance_graph(edges.flip(0),num_classes=1)
    torch.testing.assert_close(re,e);torch.testing.assert_close(ra,a)
    # Relabel objects without changing geometry or overlap counts.
    relabeled=cls(torch.tensor([0,1,3,4]),torch.tensor([9,9,5,5]),torch.tensor([4,3,1,4]),torch.tensor([0,0,0,0]))
    _,renamed=relabeled.instance_graph(edges,num_classes=1)
    torch.testing.assert_close(renamed,a)
    print(json.dumps(dict(status='OFFICIAL_INSTANCE_AFFINITY_SYNTHETIC_PASS',commit=COMMIT,
        edges=e.tolist(),smooth=a.tolist(),hard=hard.tolist(),source_sha256=READS,
        executed_definitions=EXECUTED,optimizer_steps=0,model_forward=False,partition_executed=False)))

if __name__=='__main__':main()
