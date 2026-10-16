"""One bounded, cached build of the reading PDF, Kindle EPUB and paperback files."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / '.tmp/book-publication'
STATE = ROOT / 'book/research/export-state.json'
VERSION = '5.3.0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes():
    paths = [ROOT / name for name in [
        'book/planning/manuscript-plan.json', 'book/research/bibliography.json',
        'book/marketing/amazon-listing.md', 'public/book/shadow-futures-cover.png',
        'book/tools/build_reading_manuscript.py', 'book/tools/manuscript_status.py',
        'book/tools/build_publication.py', 'book/tools/build_kdp_uploads.py',
        'book/tools/verify_publication.py', 'book/tools/refresh_book.py',
        'scripts/refresh-book.mjs',
    ]]
    paths += [p for p in (ROOT / 'book/manuscript').glob('*.md') if p.name != '90-references.md']
    paths += list((ROOT / 'book/research').glob('*source-notes.json'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def run(command, label, timeout=120):
    print(label, flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{label} timed out. Check runtime access; do not repeatedly rebuild unchanged inputs.') from exc
    output = result.stdout + result.stderr
    with (TEMP / 'refresh.log').open('a', encoding='utf-8') as log:
        log.write('\n' + label + '\n' + output)
    if result.returncode:
        raise RuntimeError(f'{label} failed:\n' + '\n'.join(output.splitlines()[-16:]))
    return output


def validator():
    java = shutil.which('java')
    if not java:
        raise RuntimeError('EPUBCheck needs Java. Install Java or add it to PATH before exporting.')
    jar = Path(os.environ.get('EPUBCHECK_JAR', TEMP / f'epubcheck/epubcheck-{VERSION}/epubcheck.jar'))
    if not jar.exists():
        if os.environ.get('EPUBCHECK_JAR'):
            raise FileNotFoundError(jar)
        print(f'Caching official EPUBCheck {VERSION} (one-time download).', flush=True)
        archive_path = TEMP / f'epubcheck-{VERSION}.zip'
        url = f'https://github.com/w3c/epubcheck/releases/download/v{VERSION}/epubcheck-{VERSION}.zip'
        with urllib.request.urlopen(url, timeout=45) as response, archive_path.open('wb') as output:
            shutil.copyfileobj(response, output)
        target = TEMP / 'epubcheck'
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Unsafe validator archive path')
            archive.extractall(target)
    return java, jar


def preview(phrases):
    if not phrases:
        return []
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(str(ROOT / 'public/book/shadow-futures.pdf'))
    wanted = {' '.join(p.split()).casefold() for p in phrases}
    pages = set()
    for index in range(len(document)):
        page = document[index]
        textpage = page.get_textpage()
        text = ' '.join(textpage.get_text_range().split()).casefold()
        for phrase in list(wanted):
            if phrase in text:
                pages.update([index, min(index + 1, len(document) - 1)])
                wanted.remove(phrase)
        textpage.close()
        page.close()
        if not wanted:
            break
    if wanted:
        raise ValueError(f'Preview phrases not found: {sorted(wanted)}')
    paths = []
    for index in sorted(pages):
        path = TEMP / f'refresh-preview-page-{index+1}.png'
        page = document[index]
        page.render(scale=1.4).to_pil().save(path)
        page.close()
        paths.append(path.relative_to(ROOT).as_posix())
    document.close()
    return paths


def output_hashes():
    paths = [ROOT / name for name in [
        'book/drafts/Shadow_Futures_Complete_Manuscript.md', 'book/MANUSCRIPT.md',
        'book/manuscript/90-references.md', 'book/research/manuscript-progress.json',
        'book/research/reading-manuscript-manifest.json', 'book/research/publication-manifest.json',
        'book/research/kdp-upload-manifest.json', 'public/book/edition.json',
        'public/book/shadow-futures.pdf', 'public/book/shadow-futures-kindle-cover.jpg',
    ]]
    paths += list((ROOT / 'output/kdp').glob('*'))
    paths += list((ROOT / 'output/pdf').glob('shadow-futures*'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Rebuild even if source and output hashes are unchanged.')
    parser.add_argument('--preview', action='append', default=[], metavar='PHRASE',
                        help='Render the first PDF page containing this phrase and its following page.')
    args = parser.parse_args()
    start = time.perf_counter()
    TEMP.mkdir(parents=True, exist_ok=True)
    inputs = input_hashes()
    previous = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    unchanged = (previous.get('inputs') == inputs and previous.get('outputs') and
                 all((ROOT / name).exists() and digest(ROOT / name) == sha
                     for name, sha in previous['outputs'].items()))
    if unchanged and not args.force:
        print('Unchanged: all deliverables match the last validated export. No rebuild or repeated checks.')
        for path in preview(args.preview):
            print('Preview: ' + path)
        return
    # Fail on missing dependencies before replacing any deliverable.
    import reportlab, pypdf, pypdfium2, PIL  # noqa: F401
    java, jar = validator()
    (TEMP / 'refresh.log').write_text('', encoding='utf-8')
    for name, label in [
        ('build_reading_manuscript.py', '1/5 Assemble manuscript and check word/note budgets'),
        ('build_publication.py', '2/5 Typeset website PDF, paperback interior and cover'),
        ('build_kdp_uploads.py', '3/5 Build reflowable Kindle EPUB and KDP upload bundle'),
    ]:
        run([sys.executable, str(ROOT / 'book/tools' / name)], label)
    report = TEMP / 'epubcheck-report.json'
    run([java, '-jar', str(jar), str(ROOT / 'output/kdp/shadow-futures-kindle.epub'),
         '-j', str(report)], '4/5 Validate final Kindle EPUB with EPUBCheck', timeout=45)
    messages = json.loads(report.read_text(encoding='utf-8')).get('messages', [])
    epub_result = {'version': VERSION, 'errors': sum(m.get('severity') in ['ERROR', 'FATAL'] for m in messages),
                   'warnings': sum(m.get('severity') == 'WARNING' for m in messages),
                   'sha256': digest(ROOT / 'output/kdp/shadow-futures-kindle.epub')}
    assert epub_result['errors'] == 0, messages
    result = run([sys.executable, str(ROOT / 'book/tools/verify_publication.py')],
                 '5/5 Verify PDF text coverage, geometry, embedded fonts and links')
    pdf_result = json.loads(result)
    manifest_path = ROOT / 'book/research/kdp-upload-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['epubcheck'] = epub_result
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    previews = preview(args.preview)
    progress = json.loads((ROOT / 'book/research/manuscript-progress.json').read_text(encoding='utf-8'))
    state = {'inputs': inputs, 'outputs': output_hashes(), 'epubcheck': epub_result,
             'pdf_integrity': pdf_result, 'previews_generated': previews,
             'elapsed_seconds': round(time.perf_counter() - start, 2),
             'boundary': 'Mechanical export checks. Inspect changed-page previews; no scholarly or KDP acceptance certification.'}
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f"Ready in {state['elapsed_seconds']}s: {pdf_result['pages']}-page reading PDF, Kindle EPUB, paperback files and ZIP.")
    print(f"Main text {progress['main_words']:,}/80,000; back matter {progress['back_matter_words']:,}/15,000.")
    print(f"EPUBCheck: {epub_result['errors']} errors, {epub_result['warnings']} warnings. Output: output/kdp/shadow-futures-kdp-upload.zip")
    for path in previews:
        print('Preview: ' + path)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Book refresh stopped: {error}', file=sys.stderr)
        sys.exit(1)
