import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV281ReleaseReadiness(unittest.TestCase):
    def test_i18n_has_no_duplicate_literal_keys_and_covers_all_tr_calls(self):
        i18n_path = ROOT / 'ui' / 'i18n.py'
        tree = ast.parse(i18n_path.read_text(encoding='utf-8'))
        language_dicts = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = set()
            literal_keys = []
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self.assertNotIn(key.value, seen, f'duplicate i18n key: {key.value}')
                    seen.add(key.value)
                    literal_keys.append(key.value)
            if len(literal_keys) > 100:
                language_dicts.append(set(literal_keys))
        self.assertGreaterEqual(len(language_dicts), 2)

        used = set()
        pattern = re.compile(r'\btr\(["\']([^"\']+)["\']\)')
        for path in ROOT.rglob('*.py'):
            if '__pycache__' not in path.parts:
                used.update(pattern.findall(path.read_text(encoding='utf-8', errors='ignore')))
        for keys in language_dicts[:2]:
            self.assertFalse(used - keys, f'missing translation keys: {sorted(used - keys)}')

    def test_validation_script_has_no_session_specific_paths(self):
        source = (ROOT / 'validate_effect_controls.py').read_text(encoding='utf-8')
        self.assertIn('argparse.ArgumentParser', source)
        forbidden_root = '/' + 'mnt' + '/data/'
        self.assertNotIn(forbidden_root, source)
        self.assertNotIn('PAGE_PATHS', source)

    def test_docs_match_current_release(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('Colortina V2.8', readme)
        self.assertIn('点采集（邻域中值）', readme)
        self.assertIn('区域采集（区域中值）', readme)
        self.assertNotIn('Colortina v5', readme)
        self.assertNotIn('“上色 / 参考 / 编辑', readme)

    def test_gitignore_covers_local_generated_data(self):
        ignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
        for token in ('runtime/', '.venv/', 'models/weights/', '*.ccproject', '*_assets/'):
            self.assertIn(token, ignore)


if __name__ == '__main__':
    unittest.main()
