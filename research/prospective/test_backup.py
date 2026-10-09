"""Backups retain a verifiable chain and reject altered or empty registries."""
from pathlib import Path
import tempfile
import unittest
import zipfile
from .registry import Registro
from .backup_registry import backup


class BackupTests(unittest.TestCase):
    def test_export_restore_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = Registro(root/'registry')
            r.iniciar(dict(primeiro_mes='2099-01', ultimo_mes='2099-12', fontes={'nino34': {}}))
            r.recibo('nino34', b'original data', dict(validado=True, meses=['2098-10']), {})
            before = r.eventos(conferir_objetos=True)
            result = backup(r.pasta, root/'snapshot.zip')
            with zipfile.ZipFile(root/'snapshot.zip') as z:
                z.extractall(root/'restored')  # Only the archive created by this test.
            self.assertEqual(Registro(root/'restored/registry').eventos(conferir_objetos=True), before)
            self.assertEqual(r.eventos(conferir_objetos=True), before)
            self.assertEqual(result['event_count'], 2)
            self.assertEqual(result['referenced_objects_verified'], 1)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                backup(r.pasta, root/'snapshot.zip')
            next((r.pasta/'objetos').glob('*.bin')).write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'Modified object'):
                backup(r.pasta, root/'bad.zip')
            self.assertFalse((root/'bad.zip').exists())

    def test_nonexistent_registry_does_not_create_an_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'existing registry'):
                backup(root/'missing', root/'snapshot.zip')
            self.assertFalse((root/'missing').exists())


if __name__ == '__main__':
    unittest.main()
