"""Check publication geometry, source coverage, fonts and working PDF links."""
from pathlib import Path
import hashlib
import json
import re

import pypdfium2 as pdfium
from pypdf import PdfReader
from build_reading_manuscript import validate_reader_citations

ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(text):return re.sub('[^a-z]','',text.lower())
def plain(text):
    text=re.sub(r'\[\^[^\]]+\](?::\s*)?','',text)
    return re.sub(r'\[([^\]]+)\]\([^\n)]+\)',r'\1',text)


def verify():
    m=json.loads((ROOT/'book/research/publication-manifest.json').read_text(encoding='utf-8'))
    path=ROOT/'public/book/shadow-futures.pdf'
    assert digest(path)==m['pdf_sha256']
    assert digest(ROOT/'book/drafts/Shadow_Futures_Complete_Manuscript.md')==m['manuscript_sha256']
    for i in m['inputs']:assert digest(ROOT/i['file'])==i['sha256'],i['file']
    reader=PdfReader(path);assert len(reader.pages)==m['pages']
    assert m['interior_pages']%2==0
    for p in reader.pages:
        assert abs(float(p.mediabox.width)-432)<.01 and abs(float(p.mediabox.height)-648)<.01
    document=pdfium.PdfDocument(str(path));texts=[]
    for i in range(1,len(document)):
        page=document[i];tp=page.get_textpage()
        texts.append(tp.get_text_bounded(left=50,bottom=51,right=382,top=597))
        tp.close();page.close()
    document.close();normalized=normalize(' '.join(texts))
    validate_reader_citations(re.sub(r'\s+', ' ', ' '.join(texts)))
    missing=[];checked=0
    for i in m['inputs']:
        text=(ROOT/i['file']).read_text(encoding='utf-8-sig')
        for block in re.split(r'\n\s*\n',text.strip()):
            if block.startswith('#') or block=='---':continue
            target=normalize(plain(block))
            if not target:continue
            checked+=1
            if target not in normalized:missing.append({'file':i['file'],'text':block[:120]})
    assert not missing,missing[:8]
    page_refs={p.indirect_reference.idnum for p in reader.pages}
    internal=external=0;fonts={}
    for p in reader.pages:
        for annotation in p.get('/Annots',[]):
            a=annotation.get_object();dest=a.get('/Dest')
            if isinstance(dest,list):
                internal+=1;assert getattr(dest[0],'idnum',None) in page_refs
            if a.get('/A',{}).get('/URI'):external+=1
        for f in p.get('/Resources',{}).get('/Font',{}).values():
            f=f.get_object();desc=f.get('/FontDescriptor',{})
            desc=desc.get_object() if hasattr(desc,'get_object') else desc
            fonts[str(f.get('/BaseFont'))]=bool(desc.get('/FontFile2') or desc.get('/FontFile3') or desc.get('/FontFile'))
    assert internal>=394 and external>=150
    assert all(fonts.values()),fonts
    result={'status':'passed','pages':m['pages'],'source_paragraphs_present_in_full':checked,
            'internal_links':internal,'external_links':external,'fonts_embedded':True,
            'pdf_sha256':m['pdf_sha256'],'boundary':'Text/geometry/link integrity, not source truth or print approval.'}
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':verify()
