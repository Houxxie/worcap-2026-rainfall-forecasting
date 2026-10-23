"""Run the bounded GitHub check suite locally, without training or source access."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['scripts/verify_repository.py'],
    ['demo/explore.py', '--check'],
    ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    ['research/prospective/test_registry.py', '-v'],
    ['-m', 'unittest', 'research.prospective.test_backup',
     'research.prospective.test_cfsv2_recipe', 'research.prospective.test_cfsv2_compatibility',
     'research.workbench.test_workbench', '-v'],
    ['research/prospective/test_verification.py', '-v'],
]


if __name__ == '__main__':
    for command in COMMANDS:
        print('\nCheck: python ' + ' '.join(command), flush=True)
        result = subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    print('\nAll repository and synthetic-data checks passed. No forecast was trained or issued.')
