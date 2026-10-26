import os
import subprocess
import sys
from pathlib import Path


PROJECT=Path(__file__).resolve().parents[1]
PYTHON=PROJECT/".venv"/"Scripts"/"python.exe"


def _python(code,*args):
    return subprocess.run([str(PYTHON),"-c",code,*map(str,args)],cwd=PROJECT,capture_output=True,text=True)


def test_isolation_paths_are_set_before_storage_modules_initialize(tmp_path):
    script="""
import sys
import benchmark
root=sys.argv[1]
paths=benchmark._configure_isolation(root)
assert not ({'config','database','mem0_engine','hybrid_retriever'} & set(sys.modules))
runtime=benchmark._load_runtime(paths)
import config, mem0_engine
assert config.settings.database_path == str(paths.db_path)
assert config.settings.mem0_path == str(paths.mem0_path)
assert runtime.db.path == str(paths.db_path)
expected_mem0 = paths.mem0_ollama_path if config.settings.provider == 'ollama' else paths.mem0_path
assert runtime.mem0_backend_path == expected_mem0
assert runtime.mem0_backend_path.is_relative_to(paths.root)
assert __import__('os').environ['MEM0_DIR'] == str(paths.mem0_config)
assert 'qdrant' in ''.join(__import__('inspect').getsource(mem0_engine._memory).split()).lower()
assert mem0_engine._memory.cache_info().currsize == 0
print('isolated configuration verified')
"""
    result=_python(script,tmp_path/"run")
    assert result.returncode==0, result.stderr
    assert "isolated configuration verified" in result.stdout


def test_setup_refuses_preimported_storage_before_side_effects(tmp_path):
    script="""
import sys
import benchmark
from pathlib import Path
root=Path(sys.argv[1])
sys.modules['database']=object()
try:
    benchmark._configure_isolation(root)
except benchmark.IsolationError:
    assert not root.exists()
else:
    raise AssertionError('unsafe setup was accepted')
print('failed closed before storage setup')
"""
    result=_python(script,tmp_path/"refused")
    assert result.returncode==0, result.stderr


def test_setup_rejects_ollama_store_and_mem0_config_collisions(tmp_path):
    script="""
import os,sys
from pathlib import Path
import benchmark
base=Path(sys.argv[1])
for name in ('ollama-store','mem0-config'):
    root=base/name
    os.environ['CAUSALMEM_DB']=str(base/'application.sqlite3')
    os.environ['MEM0_PATH']=str(base/'application-mem0')
    os.environ['MEM0_DIR']=str(base/'application-mem0-config')
    if name=='ollama-store':
        os.environ['MEM0_PATH']=str(root/'storage'/'mem0-ollama')
    else:
        os.environ['MEM0_DIR']=str(root/'storage'/'mem0-config')
    try:
        benchmark._configure_isolation(root)
    except benchmark.IsolationError:
        assert not root.exists()
    else:
        raise AssertionError(f'collision was accepted: {name}')
print('Mem0 backend collisions rejected')
"""
    result=_python(script,tmp_path/"collisions")
    assert result.returncode==0, result.stderr


def test_output_and_generated_dataset_use_requested_isolated_paths(tmp_path):
    script="""
import sys
import benchmark
from dataset_generator import generate
root=sys.argv[1]
paths=benchmark._configure_isolation(root)
out=benchmark._resolve_evaluation_paths(paths,output_dir=str(paths.root/'results'),generate=True)
written=generate(out.dataset_path)
assert written == out.dataset_path
assert out.dataset_path.is_relative_to(paths.root)
assert out.result_path == out.output_dir/'benchmark_results.json'
try:
    generate(out.dataset_path)
except FileExistsError:
    pass
else:
    raise AssertionError('dataset generator overwrote existing data')
from evaluator import save
save([],out.result_path)
assert out.result_path.is_file() and out.result_path.is_relative_to(paths.root)
print('dataset and results paths verified')
"""
    result=_python(script,tmp_path/"outputs")
    assert result.returncode==0, result.stderr


def test_setup_failure_does_not_call_ingestion_or_memory_writes(tmp_path):
    script="""
import sys
import benchmark
from pathlib import Path
calls=[]
sys.modules['mem0_engine']=object()
benchmark._load_runtime=lambda isolation: calls.append('runtime-loaded')
benchmark._run_dataset=lambda *args: calls.append('dataset-run')
try:
    benchmark.main(['--isolation-dir',sys.argv[1]])
except SystemExit as exc:
    assert exc.code==2
    pass
else:
    raise AssertionError('unsafe benchmark setup was accepted')
assert calls==[], 'runtime loading or benchmark execution followed setup failure'
assert not Path(sys.argv[1]).exists()
print('no runtime, ingestion, or memory operations followed setup failure')
"""
    result=_python(script,tmp_path/"no-writes")
    assert result.returncode==0, result.stderr


def test_benchmark_setup_does_not_change_parent_environment(tmp_path):
    names=("CAUSALMEM_DB","MEM0_PATH","MEM0_DIR")
    before={name:os.environ.get(name) for name in names}
    result=_python("import benchmark,sys; benchmark._configure_isolation(sys.argv[1])",tmp_path/"child-env")
    assert result.returncode==0, result.stderr
    assert {name:os.environ.get(name) for name in names}==before
