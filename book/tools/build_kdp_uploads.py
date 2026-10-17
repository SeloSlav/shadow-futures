"""Package a reflowable Kindle EPUB and the current paperback upload files.

Run after build_reading_manuscript.py and build_publication.py. No prose is edited.
"""
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
import hashlib
import html
import json
import posixpath
import re
import shutil
import uuid
from build_reading_manuscript import validate_reader_citations

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/kdp'
TITLE = 'Shadow Futures'
SUBTITLE = 'How Markets Choose Winners—and Lose Sight of the Alternatives'
AUTHOR = 'Martin Erlic'
XHTML = 'http://www.w3.org/1999/xhtml'
EPUB = 'http://www.idpf.org/2007/ops'
CSS = '''body { font-family: serif; line-height: 1.45; }
p { margin: 0; text-indent: 1.1em; orphans: 2; widows: 2; }
h1, h2, h3 { font-family: sans-serif; text-align: left; line-height: 1.2; }
h1 { margin: 1.8em 0 1em; } h2 { margin: 1.6em 0 .7em; }
h1 + p, h2 + p, h3 + p, p.first, aside p { text-indent: 0; }
.eyebrow { font-family: sans-serif; text-indent: 0; margin-top: 2em; }
.titlepage { text-align: center; } .titlepage h1 { text-align: center; }
.titlepage p { text-indent: 0; margin: 1.2em 0; }
.cover { margin: 0; text-align: center; } .cover img { max-width: 100%; height: auto; }
aside { margin: 0 0 1em; } .reference { text-indent: 0; margin: 0 0 1em; }
a { overflow-wrap: break-word; } a.noteref { text-decoration: none; }
nav ol { padding-left: 1.5em; } nav li { margin-bottom: .5em; }
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inline(text, note_map=None, prefix='', references=None):
    tokens = []

    def stash(value):
        tokens.append(value)
        return f'ZZEPUBTOKEN{len(tokens)-1}ZZ'

    # Scan balanced destinations so parentheses in source URLs remain intact.
    pattern = re.compile(r'\[([^\]\n]+)\]\(')
    parts = []
    cursor = 0
    while match := pattern.search(text, cursor):
        end = match.end()
        depth = 1
        while end < len(text) and depth:
            if text[end] == '(':
                depth += 1
            elif text[end] == ')':
                depth -= 1
            end += 1
        if depth:
            break
        parts.append(text[cursor:match.start()])
        label, url = match.group(1), text[match.end():end-1]
        if url.startswith(('https://', 'http://')):
            value = f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'
        else:
            # Local links have no public download target.
            value = html.escape(label)
        parts.append(stash(value))
        cursor = end
    parts.append(text[cursor:])
    text = ''.join(parts)

    def note(match):
        key = match.group(1)
        number = note_map[key]
        dest = f'{prefix}-{key}'
        occurrence = references.get(dest, 0) + 1
        references[dest] = occurrence
        return stash(f'<a class="noteref" epub:type="noteref" id="ref-{dest}-{occurrence}" '
                     f'href="notes.xhtml#note-{dest}"><sup>{number}</sup></a>')

    if note_map is not None:
        text = re.sub(r'\[\^([^\]]+)\]', note, text)
    text = html.escape(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    for index, token in enumerate(tokens):
        text = text.replace(f'ZZEPUBTOKEN{index}ZZ', token)
    return text


def document(title, body, kind='bodymatter'):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
            f'<html xmlns="{XHTML}" xmlns:epub="{EPUB}" lang="en" xml:lang="en">'
            f'<head><title>{html.escape(title)}</title><link rel="stylesheet" '
            'type="text/css" href="style.css"/></head>'
            f'<body epub:type="{kind}">{body}</body></html>')


def paragraphs(body, note_map=None, prefix='', references=None, back=False):
    output = []
    first = True
    for block in re.split(r'\n\s*\n', body.strip()):
        block = block.strip()
        if not block or block.startswith('# ') or block == '---':
            continue
        heading = re.match(r'^(#{2,3})\s+(.*)', block, re.S)
        if heading:
            level = len(heading.group(1))
            output.append(f'<h{level}>{inline(heading.group(2))}</h{level}>')
            first = True
            continue
        cls = 'reference' if back else 'first' if first else 'body'
        text = inline(' '.join(block.splitlines()), note_map, prefix, references)
        output.append(f'<p class="{cls}">{text}</p>')
        first = False
    return '\n'.join(output)


def verify_epub(path, plan):
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        assert archive.infolist()[0].filename == 'mimetype'
        assert archive.infolist()[0].compress_type == ZIP_STORED
        assert archive.read('mimetype') == b'application/epub+zip'
        trees = {}
        for name in sorted(names):
            if name.endswith(('.xml', '.opf', '.ncx', '.xhtml')):
                trees[name] = ET.fromstring(archive.read(name))
        id_sets = {}
        for name, tree in trees.items():
            identifiers = [e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
            assert len(identifiers) == len(set(identifiers)), name
            id_sets[name] = set(identifiers)
        internal_links = 0
        for name, tree in trees.items():
            for element in tree.iter():
                for attribute in ['href', 'src']:
                    target = element.get(attribute)
                    if target is None or urlsplit(target).scheme:
                        continue
                    parts = urlsplit(target)
                    target_name = (posixpath.normpath(str(PurePosixPath(name).parent / unquote(parts.path)))
                                   if parts.path else name)
                    assert target_name in names, (name, target)
                    if parts.fragment:
                        assert unquote(parts.fragment) in id_sets[target_name], (name, target)
                    internal_links += 1
        # Each source paragraph must survive conversion, including its endnotes.
        text = ' '.join(' '.join(tree.itertext()) for name, tree in trees.items() if name.endswith('.xhtml'))
        validate_reader_citations(text)
        normalize = lambda value: re.sub('[^a-z]', '', value.lower())
        plain = normalize(text)
        checked = 0
        for item in plan['components'] + plan['back_matter']:
            source = (ROOT / item['file']).read_text(encoding='utf-8-sig')
            for block in re.split(r'\n\s*\n', source.strip()):
                if block.startswith('#') or block == '---':
                    continue
                block = re.sub(r'\[\^[^\]]+\](?::\s*)?', '', block)
                rendered = inline(' '.join(block.splitlines()))
                value = normalize(' '.join(ET.fromstring(f'<div xmlns="{XHTML}">{rendered}</div>').itertext()))
                assert value in plain, (item['file'], block[:100])
                checked += 1
        assert 'OpenAI Codex assisted' not in text
        assert 'inclusion is not a claim of independent endorsement' not in text
        return {'source_paragraphs_present_in_full': checked, 'internal_links_checked': internal_links,
                'xml_files_checked': len(trees), 'status': 'passed'}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'book/planning/manuscript-plan.json').read_text(encoding='utf-8'))
    publication = json.loads((ROOT / 'book/research/publication-manifest.json').read_text(encoding='utf-8'))
    assert sha(ROOT / 'book/drafts/Shadow_Futures_Complete_Manuscript.md') == publication['manuscript_sha256']
    for item in publication['inputs']:
        assert sha(ROOT / item['file']) == item['sha256'], item['file']
    files = {'style.css': CSS}
    files['cover.xhtml'] = document('Cover', '<div class="cover"><img src="cover.jpg" '
        f'alt="Cover: {html.escape(TITLE + ': ' + SUBTITLE + ', by ' + AUTHOR, quote=True)}"/></div>', 'cover')
    files['title.xhtml'] = document(TITLE, f'<section class="titlepage"><h1>{TITLE}</h1>'
        f'<p>{html.escape(SUBTITLE)}</p><p>{AUTHOR}</p></section>'
        f'<section><h2>Copyright</h2><p class="first">Copyright © 2026 {AUTHOR}. All rights reserved.</p>'
        '<p class="first">First edition · October 2026</p></section>', 'frontmatter')
    contents = [('title.xhtml', 'Title and copyright')]
    all_notes = []
    references = {}
    counter = 0
    for index, item in enumerate(plan['components']):
        raw = (ROOT / item['file']).read_text(encoding='utf-8-sig')
        body, _, notes = raw.partition('\n## Notes\n')
        definitions = re.findall(r'^\[\^([^\]]+)\]:\s*(.*?)(?=\n\[\^|\Z)', notes, re.M | re.S)
        definitions.sort(key=lambda pair: body.index(f'[^{pair[0]}]'))
        note_map = {}
        for key, note_text in definitions:
            counter += 1
            note_map[key] = counter
            all_notes.append((item, key, counter, note_text.strip()))
        label = ('Prologue' if index == 0 else 'Introduction' if index == 1 else
                 'Epilogue' if index == 14 else f'Chapter {index-1}')
        body = re.sub(r'^## ' + re.escape(item['title']) + r'\s*$', '', body, count=1, flags=re.M)
        chapter = (f'<section epub:type="chapter"><p class="eyebrow">{label}</p>'
                   f'<h1>{html.escape(item["title"])}</h1>'
                   f'{paragraphs(body, note_map, item["id"], references)}</section>')
        filename = item['id'] + '.xhtml'
        files[filename] = document(item['title'], chapter)
        contents.append((filename, label + ': ' + item['title']))
    notes_html = ['<section epub:type="endnotes"><h1>Notes</h1>']
    previous = None
    for item, key, number, text in all_notes:
        if previous != item['id']:
            notes_html.append('<h2>' + html.escape(item['title']) + '</h2>')
            previous = item['id']
        dest = item['id'] + '-' + key
        assert references.get(dest), dest
        notes_html.append(f'<aside epub:type="endnote" id="note-{dest}"><p>'
            f'<a href="{item["id"]}.xhtml#ref-{dest}-1">{number}.</a> '
            f'{inline(" ".join(text.splitlines()))}</p></aside>')
    notes_html.append('</section>')
    files['notes.xhtml'] = document('Notes', '\n'.join(notes_html), 'backmatter')
    contents.append(('notes.xhtml', 'Notes'))
    for item in plan['back_matter']:
        label = 'References' if item['id'] == 'references' else 'Acknowledgments'
        raw = (ROOT / item['file']).read_text(encoding='utf-8-sig')
        filename = item['id'] + '.xhtml'
        files[filename] = document(label, f'<h1>{label}</h1>' + paragraphs(raw, back=True), 'backmatter')
        contents.append((filename, label))
    nav = '<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>' + ''.join(
        f'<li><a href="{name}">{html.escape(label)}</a></li>' for name, label in contents) + '</ol></nav>'
    nav += ('<nav epub:type="landmarks" hidden="hidden"><h2>Guide</h2><ol>'
            '<li><a epub:type="cover" href="cover.xhtml">Cover</a></li>'
            '<li><a epub:type="toc" href="nav.xhtml#toc">Contents</a></li>'
            '<li><a epub:type="bodymatter" href="prologue.xhtml">Start reading</a></li></ol></nav>')
    files['nav.xhtml'] = document('Contents', nav, 'frontmatter')
    identifier = 'urn:uuid:' + str(uuid.uuid5(uuid.NAMESPACE_URL, 'https://shadow-futures.vercel.app/book'))
    files['toc.ncx'] = ('<?xml version="1.0" encoding="UTF-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f'<head><meta name="dtb:uid" content="{identifier}"/></head><docTitle><text>{TITLE}</text></docTitle>'
        '<navMap>' + ''.join(f'<navPoint id="nav-{i}" playOrder="{i}"><navLabel><text>{html.escape(label)}</text>'
        f'</navLabel><content src="{name}"/></navPoint>' for i, (name, label) in enumerate(contents, 1)) + '</navMap></ncx>')
    manifest_items = []
    for index, name in enumerate(files):
        media = 'text/css' if name.endswith('.css') else 'application/x-dtbncx+xml' if name.endswith('.ncx') else 'application/xhtml+xml'
        props = ' properties="nav"' if name == 'nav.xhtml' else ''
        item_id = 'ncx' if name == 'toc.ncx' else f'f{index}'
        manifest_items.append(f'<item id="{item_id}" href="{name}" media-type="{media}"{props}/>')
    manifest_items.append('<item id="cover-image" href="cover.jpg" media-type="image/jpeg" properties="cover-image"/>')
    ordered = ['cover.xhtml', 'title.xhtml', 'nav.xhtml'] + [name for name, _ in contents if name != 'title.xhtml']
    file_ids = {name: f'f{index}' for index, name in enumerate(files)}
    opf = ('<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id" '
        'prefix="schema: http://schema.org/">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:identifier id="book-id">{identifier}</dc:identifier><dc:title>{html.escape(TITLE + ": " + SUBTITLE)}</dc:title>'
        f'<dc:creator>{AUTHOR}</dc:creator><dc:language>en</dc:language><dc:date>2026-10-16</dc:date>'
        '<meta property="dcterms:modified">2026-10-16T00:00:00Z</meta>'
        '<meta name="cover" content="cover-image"/>'
        '<meta property="schema:accessMode">textual</meta><meta property="schema:accessMode">visual</meta>'
        '<meta property="schema:accessModeSufficient">textual</meta>'
        '<meta property="schema:accessibilityFeature">tableOfContents</meta>'
        '<meta property="schema:accessibilityFeature">structuralNavigation</meta>'
        '<meta property="schema:accessibilityFeature">displayTransformability</meta>'
        '<meta property="schema:accessibilityHazard">none</meta></metadata>'
        '<manifest>' + ''.join(manifest_items) + '</manifest><spine toc="ncx">' + ''.join(
            f'<itemref idref="{file_ids[name]}"/>' for name in ordered) + '</spine></package>')
    target = OUT / 'shadow-futures-kindle.epub'
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('mimetype', 'application/epub+zip', compress_type=ZIP_STORED)
        archive.writestr('META-INF/container.xml', '<?xml version="1.0"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        for name, text in files.items():
            archive.writestr('OEBPS/' + name, text.encode('utf-8'))
        archive.writestr('OEBPS/content.opf', opf.encode('utf-8'))
        archive.write(ROOT / 'public/book/shadow-futures-kindle-cover.jpg', 'OEBPS/cover.jpg')
    checks = verify_epub(target, plan)
    copies = {'shadow-futures-kindle-cover.jpg': ROOT / 'public/book/shadow-futures-kindle-cover.jpg',
              'shadow-futures-paperback-interior.pdf': ROOT / 'output/pdf/shadow-futures-interior-6x9.pdf',
              'shadow-futures-paperback-cover.pdf': ROOT / 'output/pdf/shadow-futures-kdp-paperback-cover.pdf'}
    for name, source in copies.items():
        destination = OUT / name
        if destination.exists() and source.read_bytes() == destination.read_bytes():
            continue
        pending = OUT / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            shutil.copy2(source, pending)
            pending.replace(destination)
        except OSError as exc:
            raise OSError(f'Cannot refresh upload file {destination}: {exc}') from exc
        finally:
            pending.unlink(missing_ok=True)
    instructions = f'''SHADOW FUTURES — KDP UPLOAD FILES
Martin Erlic

KINDLE EBOOK
Manuscript: shadow-futures-kindle.epub (reflowable text; linked contents/endnotes)
Cover: shadow-futures-kindle-cover.jpg (1600 x 2560 JPEG)
Upload the EPUB as the manuscript and the JPEG separately as the marketing cover.

PAPERBACK
Manuscript: shadow-futures-paperback-interior.pdf
Cover: shadow-futures-paperback-cover.pdf (full back/spine/front wrap)
Choose: 6 x 9 inch trim; black ink; WHITE paper; NO BLEED for the interior.
The cover includes its required 0.125-inch outer bleed independently.
Interior: {publication['interior_pages']} pages. Spine: {publication['paperback']['spine_inches']:.6f} inches.
Full cover: {publication['paperback']['wrap_inches'][0]:.6f} x 9.25 inches.
Choose matte or glossy finish to taste; no finish-specific file change is needed.
The back leaves a clear area for KDP's barcode. No ISBN has been invented.
If adding your own ISBN, enter that actual ISBN in KDP before approving the preview.
If paper, trim or page count changes, regenerate the cover rather than reusing this spine.

REVIEW AND DISCLOSURE
Open KDP's Kindle and Print Previewers and approve the previews before publishing.
Order a print proof to assess the existing artwork: its native source is 1024 x 1536;
the print-size raster is resampled. Correct geometry does not guarantee sharp original detail.
The text and cover contain AI-generated material. Answer KDP's AI-content questions
accurately for both formats; the removed public acknowledgment is not that disclosure.
https://kdp.amazon.com/en_US/help/topic/G200672390

PUBLICATION SETTINGS
Use the exact existing title, subtitle and Martin Erlic author name on both editions.
Leave KDP Select OFF while the full digital book is available free on the website.
Select requires digital exclusivity during enrollment; ordinary KDP publication does not:
https://kdp.amazon.com/en_US/help/topic/GD9PMU58BV24QFZ7
See amazon-listing.md for the three categories, seven keyword fields and description.
EPUB is an accepted KDP manuscript format:
https://kdp.amazon.com/en_US/help/topic/G200634390/
This package does not set a price, choose rights, enter an ISBN or publish the book.
Original technical source-check limits remain disclosed in the manuscript's notes.
'''
    (OUT / 'UPLOAD-INSTRUCTIONS.txt').write_text(instructions, encoding='utf-8', newline='\n')
    shutil.copy2(ROOT / 'book/marketing/amazon-listing.md', OUT / 'amazon-listing.md')
    names = [target.name, *copies, 'UPLOAD-INSTRUCTIONS.txt', 'amazon-listing.md']
    bundle = OUT / 'shadow-futures-kdp-upload.zip'
    with ZipFile(bundle, 'w', compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(OUT / name, name)
    record = {'date': '2026-10-16', 'manuscript_sha256': publication['manuscript_sha256'],
              'epub_integrity': checks, 'endnotes': counter, 'paperback': publication['paperback'],
              'files': [{'file': str((OUT / name).relative_to(ROOT)).replace('\\', '/'),
                         'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)}
                        for name in [*names, bundle.name]],
              'boundary': 'Upload formats and mechanical checks; KDP Previewer and print proof remain author publication steps.'}
    (ROOT / 'book/research/kdp-upload-manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'epub': str(target.relative_to(ROOT)), 'endnotes': counter, **checks,
                      'bundle': str(bundle.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    build()
