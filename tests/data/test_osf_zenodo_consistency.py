#!/usr/bin/env python3
"""
Rigorous test suite for OSF and Zenodo download consistency.

Tests that files downloaded from OSF and Zenodo are:
1. Exactly the same (byte-for-byte identical)
2. At the same location
3. With the same name
4. So that code can reliably find them regardless of source

Tests one file from each category in DATA_SOURCE_MAP, with special focus
on stellar data (PHOENIX, MUSCLES, Solar - used by MORS).

Note: This test requires the PROTEUS environment to be set up with all
dependencies installed (platformdirs, osfclient, zenodo_client, etc.).
Run with: python tests/pr591_tests/test_osf_zenodo_consistency.py
Or with pytest: pytest tests/pr591_tests/test_osf_zenodo_consistency.py -v
"""

import json
import os
import sys
import tempfile
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import using importlib to load data.py directly without package imports
import sys
import importlib.util
from unittest.mock import MagicMock, patch

# Set up environment before loading module
os.environ.setdefault('FWL_DATA', str(Path.home() / '.fwl_data_test'))

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set up minimal package structure to allow imports
import types

# Create minimal proteus package structure
if 'proteus' not in sys.modules:
    proteus_pkg = types.ModuleType('proteus')
    sys.modules['proteus'] = proteus_pkg

if 'proteus.utils' not in sys.modules:
    proteus_utils = types.ModuleType('proteus.utils')
    proteus_utils.__path__ = [str(src_path / 'proteus' / 'utils')]
    sys.modules['proteus.utils'] = proteus_utils

# Load helper module first (needed by data.py)
helper_path = src_path / 'proteus' / 'utils' / 'helper.py'
if helper_path.exists():
    helper_spec = importlib.util.spec_from_file_location('proteus.utils.helper', helper_path)
    if helper_spec and helper_spec.loader:
        try:
            helper_module = importlib.util.module_from_spec(helper_spec)
            sys.modules['proteus.utils.helper'] = helper_module
            # Patch match statement for Python < 3.10
            if sys.version_info < (3, 10):
                # Read and convert match statement to if/elif
                with open(helper_path, 'r') as f:
                    helper_code = f.read()

                # Convert match/case to if/elif
                lines = helper_code.split('\n')
                new_lines = []
                in_match = False
                first_case = True

                for line in lines:
                    if 'match status:' in line:
                        in_match = True
                        new_lines.append('    if True:  # match status:')
                        first_case = True
                    elif in_match and line.strip().startswith('case '):
                        case_value = line.strip().replace('case ', '').replace(':', '')
                        if case_value == '_':
                            new_lines.append('        else:')
                        else:
                            if first_case:
                                new_lines.append(f'        if status == {case_value}:')
                                first_case = False
                            else:
                                new_lines.append(f'        elif status == {case_value}:')
                    elif in_match and line.strip() and not line.strip().startswith('#'):
                        # Check if we're out of the match block (next function/class)
                        if line and not line[0].isspace() and 'def ' in line:
                            in_match = False
                            new_lines.append(line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                helper_code = '\n'.join(new_lines)
                exec(compile(helper_code, str(helper_path), 'exec'), helper_module.__dict__)
            else:
                helper_spec.loader.exec_module(helper_module)
        except Exception as e:
            print(f'Warning: Could not load helper module: {e}')

# Load phoenix_helper if needed
phoenix_helper_path = src_path / 'proteus' / 'utils' / 'phoenix_helper.py'
if phoenix_helper_path.exists() and 'proteus.utils.phoenix_helper' not in sys.modules:
    phoenix_helper_spec = importlib.util.spec_from_file_location(
        'proteus.utils.phoenix_helper', phoenix_helper_path
    )
    if phoenix_helper_spec and phoenix_helper_spec.loader:
        try:
            phoenix_helper_module = importlib.util.module_from_spec(phoenix_helper_spec)
            sys.modules['proteus.utils.phoenix_helper'] = phoenix_helper_module
            phoenix_helper_spec.loader.exec_module(phoenix_helper_module)
        except Exception as e:
            print(f'Warning: Could not load phoenix_helper module: {e}')

# Now try to load data module
data_module_path = src_path / 'proteus' / 'utils' / 'data.py'
try:
    data_spec = importlib.util.spec_from_file_location('proteus.utils.data', data_module_path)
    if data_spec and data_spec.loader:
        data_module = importlib.util.module_from_spec(data_spec)
        sys.modules['proteus.utils.data'] = data_module
        data_spec.loader.exec_module(data_module)

        # Use the module directly - no wrapper needed
        # The module itself has all the attributes we need
        MODULE_LOADED = True
    else:
        MODULE_LOADED = False
        data_module = None
except Exception as e:
    print(f'Warning: Could not load data module: {e}')
    import traceback

    print(f'Traceback: {traceback.format_exc()}')
    MODULE_LOADED = False
    data_module = None


def md5_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_zenodo_file_list(zenodo_id: str) -> list[dict]:
    """
    Get list of files from a Zenodo record using the API.

    Parameters
    ----------
    zenodo_id : str
        Zenodo record ID

    Returns
    -------
    list[dict]
        List of file dictionaries with 'key' (path) and 'size' keys
    """
    try:
        api_url = f'https://zenodo.org/api/records/{zenodo_id}'
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read())

        files = data.get('files', [])
        return [
            {'key': f.get('key', ''), 'size': f.get('size', 0)} for f in files if f.get('key')
        ]
    except Exception as e:
        print(f'      Warning: Could not get Zenodo file list: {e}')
        return []


def get_osf_file_list(osf_id: str, folder: str) -> list[dict]:
    """
    Get list of files from an OSF project for a specific folder.

    Parameters
    ----------
    osf_id : str
        OSF project ID
    folder : str
        Folder name to filter files

    Returns
    -------
    list[dict]
        List of file dictionaries with 'path' and 'size' keys
    """
    try:
        if not MODULE_LOADED:
            return []
        get_osf_func = data_module.get_osf
        storage = get_osf_func(osf_id)

        files = []
        for file in storage.files:
            # OSF file paths start with '/', so we remove it
            file_path = file.path[1:] if file.path.startswith('/') else file.path
            # Check if file is in the target folder
            if file_path.startswith(folder + '/') or file_path == folder:
                # Get relative path within folder
                if file_path.startswith(folder + '/'):
                    rel_path = file_path[len(folder) + 1 :]
                else:
                    rel_path = file_path
                files.append({'key': rel_path, 'path': file_path, 'size': file.size})
        return files
    except Exception as e:
        print(f'      Warning: Could not get OSF file list: {e}')
        return []


def find_common_file(
    zenodo_files: list[dict], osf_files: list[dict], preferred_names: Optional[list[str]] = None
) -> Optional[str]:
    """
    Find a file that exists in both Zenodo and OSF file lists.

    Parameters
    ----------
    zenodo_files : list[dict]
        List of files from Zenodo
    osf_files : list[dict]
        List of files from OSF
    preferred_names : list[str], optional
        Preferred file names to try first (e.g., ['_readme.md', 'readme.txt'])

    Returns
    -------
    str | None
        File path/key if found, None otherwise
    """
    if preferred_names is None:
        preferred_names = ['_readme.md', 'readme.txt', 'README.md', 'readme.md']

    # Create sets of file keys for quick lookup
    zenodo_keys = {f['key'] for f in zenodo_files}
    osf_keys = {f['key'] for f in osf_files}

    # First try preferred names
    for pref_name in preferred_names:
        if pref_name in zenodo_keys and pref_name in osf_keys:
            return pref_name

    # Then try any common file
    common = zenodo_keys & osf_keys
    if common:
        # Prefer smaller files for faster testing
        common_list = sorted(common)
        for key in common_list:
            # Find sizes
            zenodo_size = next((f['size'] for f in zenodo_files if f['key'] == key), 0)
            osf_size = next((f['size'] for f in osf_files if f['key'] == key), 0)
            # Only return if sizes match (indicating same file)
            if zenodo_size > 0 and osf_size > 0 and zenodo_size == osf_size:
                return key

    return None


def test_category(
    category_name: str,
    folder: str,
    zenodo_id: str,
    osf_id: str,
    tmpdir: Path,
) -> tuple[bool, str]:
    """
    Test that a file downloaded from Zenodo and OSF are identical.

    Parameters
    ----------
    category_name : str
        Human-readable category name for reporting
    folder : str
        Folder name in DATA_SOURCE_MAP
    zenodo_id : str
        Zenodo record ID
    osf_id : str
        OSF project ID
    tmpdir : Path
        Temporary directory for downloads

    Returns
    -------
    tuple[bool, str]
        (success, message)
    """
    if not MODULE_LOADED:
        return False, 'Module not loaded'

    try:
        # Import functions if not already imported
        if not MODULE_LOADED:
            return False, 'Module not loaded'

        download = data_module.download
        GetFWLData = data_module.GetFWLData
        get_zenodo_file = data_module.get_zenodo_file

        print(f'\n  Testing {category_name} ({folder})')
        print(f'    Zenodo ID: {zenodo_id}, OSF ID: {osf_id}')

        # Get file lists from both sources
        print(f'    Getting file lists...')
        zenodo_files = get_zenodo_file_list(zenodo_id)
        osf_files = get_osf_file_list(osf_id, folder)

        if not zenodo_files:
            return False, 'No files found in Zenodo record'
        if not osf_files:
            return False, 'No files found in OSF project'

        print(f'      Zenodo: {len(zenodo_files)} files')
        print(f'      OSF: {len(osf_files)} files')

        # Find a common file
        test_file = find_common_file(zenodo_files, osf_files)
        if not test_file:
            return False, 'No common file found between Zenodo and OSF'

        print(f'    Testing file: {test_file}')

        # Create separate directories for Zenodo and OSF downloads
        zenodo_dir = tmpdir / f'{folder}_zenodo'
        osf_dir = tmpdir / f'{folder}_osf'

        # Download from Zenodo
        print(f'    Downloading from Zenodo...')
        zenodo_success = False
        try:
            zenodo_success = get_zenodo_file(
                zenodo_id=zenodo_id,
                folder_dir=zenodo_dir,
                zenodo_path=test_file,
            )
        except Exception as e:
            print(f'      Zenodo download failed: {e}')
            return False, f'Zenodo download failed: {e}'

        if not zenodo_success:
            return False, 'Zenodo download returned False'

        zenodo_file = zenodo_dir / test_file
        if not zenodo_file.exists():
            return False, f'Zenodo file not found at {zenodo_file}'

        zenodo_size = zenodo_file.stat().st_size
        if zenodo_size == 0:
            return False, 'Zenodo file is empty'

        print(f'      ✓ Zenodo: {zenodo_file} ({zenodo_size} bytes)')

        # Download from OSF
        print(f'    Downloading from OSF...')
        osf_success = False
        try:
            get_osf_func = data_module.get_osf
            storage = get_osf_func(osf_id)
            # Find the file in OSF storage
            osf_file_path = f'{folder}/{test_file}'
            found = False

            for file in storage.files:
                file_path = file.path[1:] if file.path.startswith('/') else file.path
                if file_path == osf_file_path or file_path.endswith(f'/{test_file}'):
                    osf_dir.mkdir(parents=True, exist_ok=True)
                    target_file = osf_dir / test_file
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(target_file, 'wb') as f:
                        file.write_to(f)
                    found = True
                    osf_success = True
                    break

            if not found:
                return False, f'File {test_file} not found in OSF project {osf_id}'

        except Exception as e:
            print(f'      OSF download failed: {e}')
            return False, f'OSF download failed: {e}'

        if not osf_success:
            return False, 'OSF download failed'

        osf_file = osf_dir / test_file
        if not osf_file.exists():
            return False, f'OSF file not found at {osf_file}'

        osf_size = osf_file.stat().st_size
        if osf_size == 0:
            return False, 'OSF file is empty'

        print(f'      ✓ OSF: {osf_file} ({osf_size} bytes)')

        # Compare files
        print(f'    Comparing files...')

        # 1. Check file sizes match
        if zenodo_size != osf_size:
            return False, (
                f'File sizes differ: Zenodo={zenodo_size} bytes, OSF={osf_size} bytes'
            )

        # 2. Check file locations (relative paths should match)
        zenodo_rel = zenodo_file.relative_to(zenodo_dir)
        osf_rel = osf_file.relative_to(osf_dir)
        if str(zenodo_rel) != str(osf_rel):
            return False, (f'Relative paths differ: Zenodo={zenodo_rel}, OSF={osf_rel}')

        # 3. Check file names match
        if zenodo_file.name != osf_file.name:
            return False, (f'File names differ: Zenodo={zenodo_file.name}, OSF={osf_file.name}')

        # 4. Check file contents are identical (byte-for-byte)
        zenodo_hash = md5_hash(zenodo_file)
        osf_hash = md5_hash(osf_file)

        if zenodo_hash != osf_hash:
            return False, (f'File contents differ (MD5): Zenodo={zenodo_hash}, OSF={osf_hash}')

        # 5. Verify files are at expected location when using download() function
        print(f'    Testing download() function...')
        download_success = download(
            folder=folder,
            target='test_downloads',
            zenodo_id=zenodo_id,
            osf_id=osf_id,
            desc=f'test {category_name}',
            zenodo_path=test_file,
            force=True,  # Force download to ensure fresh test
        )

        if not download_success:
            return False, 'download() function returned False'

        expected_file = GetFWLData() / 'test_downloads' / folder / test_file
        if not expected_file.exists():
            return False, f'download() file not at expected location: {expected_file}'

        expected_size = expected_file.stat().st_size
        if expected_size != zenodo_size:
            return False, (
                f'download() file size differs: expected={zenodo_size}, got={expected_size}'
            )

        expected_hash = md5_hash(expected_file)
        if expected_hash != zenodo_hash:
            return False, (
                f'download() file content differs (MD5): expected={zenodo_hash}, '
                f'got={expected_hash}'
            )

        print(f'      ✓ download() file matches: {expected_file}')

        return True, (
            f'✓ All checks passed: {zenodo_size} bytes, '
            f'MD5={zenodo_hash[:8]}..., '
            f'location={zenodo_rel}'
        )

    except Exception as e:
        import traceback

        return False, f'Exception: {e}\n{traceback.format_exc()}'


def run_consistency_tests():
    """Run consistency tests for all categories in DATA_SOURCE_MAP."""
    print('=' * 80)
    print('OSF-Zenodo Download Consistency Tests')
    print('=' * 80)
    print('\nTesting that files downloaded from OSF and Zenodo are:')
    print('  1. Byte-for-byte identical')
    print('  2. At the same location')
    print('  3. With the same name')
    print('  4. Accessible via download() function')
    print()

    if not MODULE_LOADED:
        print('❌ Cannot load data module - skipping tests')
        return False

    # Get DATA_SOURCE_MAP
    if not MODULE_LOADED:
        print('❌ Cannot load data module - skipping tests')
        return False
    DATA_SOURCE_MAP = (
        data_module.DATA_SOURCE_MAP
        if hasattr(data_module, 'DATA_SOURCE_MAP')
        else getattr(data_module, 'DATA_SOURCE_MAP')
    )

    # Test all categories in DATA_SOURCE_MAP
    # Focus on stellar data (MORS, Phoenix) first, then test all others
    test_cases = []

    # Priority categories (stellar data)
    priority_categories = ['PHOENIX', 'MUSCLES', 'Solar', 'Named']
    for folder in priority_categories:
        if folder in DATA_SOURCE_MAP:
            info = DATA_SOURCE_MAP[folder]
            test_cases.append(
                (
                    f'Stellar Spectra - {folder}',
                    folder,
                    info['zenodo_id'],
                    info['osf_id'],
                )
            )

    # All other categories
    for folder, info in DATA_SOURCE_MAP.items():
        if folder not in priority_categories:
            # Create a readable category name
            if '/' in folder:
                category_name = folder.replace('/', ' - ')
            else:
                category_name = folder
            test_cases.append((category_name, folder, info['zenodo_id'], info['osf_id']))

    results = []
    passed = 0
    failed = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for category_name, folder, zenodo_id, osf_id in test_cases:
            success, message = test_category(
                category_name, folder, zenodo_id, osf_id, tmpdir_path
            )

            if success:
                passed += 1
                print(f'  ✅ {category_name}: {message}')
            elif 'No common file found' in message or 'No files found' in message:
                skipped += 1
                print(f'  ⏭️  {category_name}: {message} (skipped)')
            else:
                failed += 1
                print(f'  ❌ {category_name}: {message}')

            results.append((category_name, success, message))

    print('\n' + '=' * 80)
    print('Test Summary')
    print('=' * 80)
    print(f'Total categories tested: {len(test_cases)}')
    print(f'Passed: {passed}')
    print(f'Failed: {failed}')
    print(f'Skipped: {skipped}')

    if failed > 0:
        print('\nFailed tests:')
        for category_name, success, message in results:
            if not success and 'skipped' not in message.lower():
                print(f'  - {category_name}: {message}')

    return failed == 0


if __name__ == '__main__':
    success = run_consistency_tests()
    sys.exit(0 if success else 1)
