"""Execute the visible TensorFlow/SavedModel/Lite CPU programs and source checks."""
import contextlib
import hashlib
import io
import json
import linecache
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md'
BEGIN = '<!-- BEGIN REVIEW: TensorFlow SavedModel 2026-10-27 -->'
END = '<!-- END REVIEW: TensorFlow SavedModel 2026-10-27 -->'


def main():
    text = CHAPTER.read_text()
    section = text.split(BEGIN, 1)[1].split(END, 1)[0]
    fences = re.findall(r'^```python\n(.*?)^```', section, re.M | re.S)
    assert len(fences) == 11, len(fences)
    assert fences[-1].strip() == 'savedmodel_temporary.cleanup()'
    ns = {'__name__': '__tensorflow_export_examples__'}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            for index, code in enumerate(fences[:-1]):
                filename = f'{CHAPTER}:tensorflow-{index+1}'
                linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
                exec(compile(code, filename, 'exec'), ns)
            tf, np = ns['tf'], ns['np']
            assert tf.__version__ == '2.21.0'
            assert not tf.config.list_physical_devices('GPU')
            np.testing.assert_allclose(ns['imported'].v.numpy(), 5-3*.6**10, atol=1e-6)
            assert ns['module'].v.numpy() == 1.
            assert ns['restored'].v.numpy() == 3.
            freshly_loaded = tf.saved_model.load(ns['servable_path'])
            assert freshly_loaded.v.numpy() == 2.
            assert not hasattr(freshly_loaded, 'python_note')
            assert set(freshly_loaded.signatures) == {'serving_default'}
            root = Path(ns['servable_path'])
            files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
            assert {'saved_model.pb','fingerprint.pb','variables/variables.index',
                    'variables/variables.data-00000-of-00001'} <= set(files)
            assert (root/'assets').is_dir()
            assert list((root/'assets').iterdir()) == []
            # A new interpreter has no CustomModule/NamedScale definition.
            child = subprocess.run([sys.executable, '-c',
                'import sys,json,tensorflow as tf; m=tf.saved_model.load(sys.argv[1]); '
                'print(json.dumps(m.signatures["serving_default"](x=tf.constant([3.]))["score"].numpy().tolist()))',
                ns['servable_path']],capture_output=True,text=True,check=True)
            assert json.loads(child.stdout) == [6.]
            rng = np.random.default_rng(20261027)
            cases = 0
            largest = 0.
            for size in [1, 2, 3, 8, 16, 31]:
                for _ in range(20):
                    x = rng.uniform(-100, 100, size).astype(np.float32)
                    saved = freshly_loaded.signatures['serving_default'](x=tf.constant(x))['score'].numpy()
                    lite = ns['lite_endpoint'](x=x)['score']
                    expected = (x*2).astype(np.float32)
                    np.testing.assert_array_equal(saved, expected)
                    np.testing.assert_array_equal(lite, expected)
                    np.testing.assert_array_equal(ns['endpoint'](x=tf.constant(x))['score'].numpy(), x*3)
                    largest = max(largest, float(np.abs(lite-saved).max()))
                    cases += 1
            for x in [tf.constant(3.), tf.constant([[3.]])]:
                assert freshly_loaded.signatures['serving_default'](x=x)['score'].shape == x.shape
                try:
                    freshly_loaded.serve(x)
                except (TypeError, ValueError):
                    pass
                else:
                    raise AssertionError('restored method accepted an unsaved input signature')
            rejected = 0
            for x in [np.array(3., np.float32), np.ones((1,1),np.float32),
                      np.ones(3,np.float64), np.ones(0,np.float32),
                      np.ones(33,np.float32), np.array([np.nan],np.float32),
                      np.array([np.inf],np.float32), [3.]]:
                try:
                    ns['checked_scale'](freshly_loaded.signatures['serving_default'], x)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError(f'application accepted bad vector: {x}')
            try:
                freshly_loaded.signatures['serving_default'](x=tf.constant([3.],tf.float64))
            except (TypeError, ValueError, tf.errors.InvalidArgumentError):
                pass
            else:
                raise AssertionError('signature executed a float64 input with float32 state')
            original = json.loads((ROOT/'05 Источники/Originals/tensorflow-export-2026-10-27/saved_model.ipynb').read_text())
            source_map = {'custom module':[31], 'save scalar':[33], 'load scalar':[36],
                          'imported training':[39,40]}
            for name, cells in source_map.items():
                excerpt = section.split(f'<!-- BEGIN SOURCE: TensorFlow {name} -->')[1].split('<!-- END SOURCE:')[0]
                code = re.findall(r'^```python\n(.*?)^```', excerpt, re.M|re.S)
                assert [c.strip() for c in code] == [''.join(original['cells'][i]['source']).strip() for i in cells]
            for name, i in [('saved functions',32),('signature selection',46),('output names',52)]:
                excerpt = section.split(f'<!-- BEGIN SOURCE: TensorFlow {name} -->\n')[1].split('\n<!-- END SOURCE:')[0]
                source = ''.join(original['cells'][i]['source']).strip()
                source = source.replace('./checkpoint.ipynb','https://www.tensorflow.org/guide/checkpoint')
                source = source.replace('(function.ipynb)','(https://www.tensorflow.org/guide/function)')
                assert excerpt == source, name
            graph_source = json.loads((ROOT/'05 Источники/Originals/tensorflow-export-2026-10-27/intro_to_graphs.ipynb').read_text())
            graph = section.split('<!-- BEGIN SOURCE: TensorFlow graph definition -->\n')[1].split('\n<!-- END SOURCE:')[0]
            assert graph in ''.join(graph_source['cells'][5]['source']).replace('**','')
            harvard = (ROOT/'05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
            exit_path = section.split('<!-- BEGIN SOURCE: Harvard framework exit path -->\n')[1].split('\n<!-- END SOURCE:')[0]
            assert exit_path in harvard
            previous = text.replace(BEGIN+section+END+'\n\n','')
            assert hashlib.sha256(previous.encode()).hexdigest() == '8d6cc94470edc9ca00b9bc2d032d009bd51a34b130bfbb49356bea3b1af2f4d8'
        finally:
            if 'savedmodel_temporary' in ns:
                exec(compile(fences[-1],str(CHAPTER)+':cleanup','exec'),ns)
    assert not Path(ns['tmpdir']).exists()
    print(json.dumps({'chapter_sha256':hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
        'tensorflow':tf.__version__,'numpy':np.__version__,'executed_fences':len(fences),
        'independent_savedmodel_lite_cases':cases,'application_rejections':rejected,
        'shape_call_paths':'restored method rejects scalar/matrix; flat signature executes them',
        'lite_vs_saved_max_error':largest,'restored_v_after_ten_steps':float(ns['imported'].v.numpy()),
        'in_memory_vs_exported_v':[3.,2.],'saved_files':files,
        'source_code_cells':5,'source_explanatory_passages':4,
        'harvard_exit_paragraph':'complete original','fresh_interpreter_without_classes':[6.],
        'prior_1088_lines':'entire revision byte-preserved','temporary_models_removed':True,
        'scope':'CPU TensorFlow save/load/gradients and Lite conversion/invoke; no Serving process, GPU, delegate, device or performance test'},indent=2))


if __name__=='__main__':
    main()
