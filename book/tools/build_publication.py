"""Typeset the current manuscript and export the free edition and KDP-size assets.

Run with the bundled Python runtime. This formats saved prose; it does not edit it.
Cover artwork is generated separately. Proof print assets before KDP submission.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import html
import json
import re
import shutil

from PIL import Image
from pypdf import PdfReader, PdfWriter
import pypdfium2 as pdfium
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
    NextPageTemplate, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf'
PUBLIC = ROOT / 'public/book'
TEMP = ROOT / '.tmp/book-publication'
TITLE = 'Shadow Futures'
SUBTITLE = 'How Markets Choose Winners—and Lose Sight of the Alternatives'
AUTHOR = 'Martin Erlic'
DATE = '2026-10-16'
PAGE = (6 * inch, 9 * inch)
INK = colors.HexColor('#182b32')
MUTED = colors.HexColor('#657075')
GOLD = colors.HexColor('#c99b55')
CREAM = colors.HexColor('#f3eee2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_fonts():
    fonts = Path('C:/Windows/Fonts')
    for name, file in [('Book','georgia.ttf'),('Book-Bold','georgiab.ttf'),
                       ('Book-Italic','georgiai.ttf'),('Book-BoldItalic','georgiaz.ttf'),
                       ('Sans','segoeui.ttf'),('Sans-Bold','segoeuib.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily('Book', normal='Book', bold='Book-Bold',
                                 italic='Book-Italic', boldItalic='Book-BoldItalic')
    pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='Sans-Bold',
                                 italic='Sans', boldItalic='Sans-Bold')
    # ReportLab uses Helvetica for some implicit labels; embed that fallback too.
    pdfmetrics.registerFont(TTFont('Helvetica', str(fonts / 'segoeui.ttf')))
    rl_config.canvas_basefontname = 'Book'


STYLES = {}


def styles():
    specs = {
        'body': dict(fontName='Book', fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                     firstLineIndent=13, spaceAfter=4, allowWidows=0, allowOrphans=0),
        'first': dict(fontName='Book', fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                      spaceAfter=4, allowWidows=0, allowOrphans=0),
        'chapter': dict(fontName='Book', fontSize=25, leading=31, spaceAfter=25,
                        keepWithNext=True),
        'section': dict(fontName='Book-Bold', fontSize=13, leading=18, spaceBefore=17,
                        spaceAfter=9, keepWithNext=True),
        'eyebrow': dict(fontName='Sans', fontSize=8, leading=12, textColor=MUTED,
                        spaceBefore=58, spaceAfter=13, keepWithNext=True),
        'note': dict(fontName='Book', fontSize=8.3, leading=11.8, spaceAfter=8,
                     allowWidows=0, allowOrphans=0, splitLongWords=True),
        'toc': dict(fontName='Book', fontSize=9.3, leading=13, leftIndent=0,
                    firstLineIndent=0, spaceBefore=0, spaceAfter=3),
        'small': dict(fontName='Book', fontSize=9, leading=13, spaceAfter=12),
        'title': dict(fontName='Book', fontSize=42, leading=47, alignment=TA_CENTER,
                      textColor=INK, spaceAfter=25),
        'subtitle': dict(fontName='Book', fontSize=15, leading=22, alignment=TA_CENTER,
                         textColor=INK, spaceAfter=45),
        'author': dict(fontName='Sans', fontSize=15, leading=21, alignment=TA_CENTER,
                       textColor=INK),
    }
    for name, spec in specs.items():
        STYLES[name] = ParagraphStyle(name, **spec)


def xml(text, note_map=None, prefix='', used=None):
    """Convert the limited Markdown used by the manuscript to PDF paragraph XML."""
    text = text.replace('\u2011', '-').replace('\u00a0', ' ')
    tokens = []
    def stash(value):
        tokens.append(value)
        return f'ZZPDFTOKEN{len(tokens)-1}ZZ'
    def link(m):
        label, url = m.groups()
        if not url.startswith(('https://', 'http://')):
            return stash(html.escape(label))
        return stash(f'<link href="{html.escape(url, quote=True)}" color="#285f73">'
                     f'{html.escape(label)}</link>')
    text = re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', link, text)
    def note(m):
        key=m.group(1);number=note_map[key];dest=f'{prefix}-{key}'
        anchor=''
        if dest not in used:
            anchor=f'<a name="ref-{dest}"/>';used.add(dest)
        return stash(f'{anchor}<super><link href="#note-{dest}" color="#285f73">{number}</link></super>')
    if note_map is not None:
        text = re.sub(r'\[\^([^\]]+)\]', note, text)
    text = html.escape(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Sans">\1</font>', text)
    for i, token in enumerate(tokens):
        text = text.replace(f'ZZPDFTOKEN{i}ZZ', token)
    return text


class BookDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path), pagesize=PAGE, leftMargin=54,rightMargin=54,
                         topMargin=54,bottomMargin=54, title=TITLE,author=AUTHOR,
                         subject=SUBTITLE, pageCompression=1)
        self.running_title=TITLE
        frame=Frame(54,54,PAGE[0]-108,PAGE[1]-108,id='text',
                    leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([
            PageTemplate('front',[frame],onPage=self.front),
            PageTemplate('body',[frame],onPage=self.running),
            PageTemplate('opener',[frame],onPage=self.opener),
        ])

    def beforeDocument(self):
        self.running_title=TITLE
        self.section_pages={}

    def front(self,c,doc):
        c.setTitle(f'{TITLE}: {SUBTITLE}');c.setAuthor(AUTHOR)

    def running(self,c,doc):
        c.saveState()
        c.setFillColor(MUTED);c.setFont('Sans',7.4)
        header=TITLE.upper() if doc.page%2==0 else self.running_title
        while pdfmetrics.stringWidth(header,'Sans',7.4)>PAGE[0]-108:
            header=header[:-2].rstrip()+'…'
        c.drawString(54,PAGE[1]-34,header)
        c.setFont('Book',9);c.drawCentredString(PAGE[0]/2,31,str(doc.page))
        c.restoreState()

    def opener(self,c,doc):
        c.saveState();c.setFillColor(MUTED);c.setFont('Book',9)
        c.drawCentredString(PAGE[0]/2,31,str(doc.page));c.restoreState()

    def afterFlowable(self,f):
        if hasattr(f,'toc_label'):
            self.running_title=f.toc_label
            self.section_pages[f.bookmark]=self.page
            self.canv.bookmarkPage(f.bookmark)
            self.canv.addOutlineEntry(f.toc_label,f.bookmark,level=0,closed=False)
            self.notify('TOCEntry',(0,f.toc_label,self.page,f.bookmark))


def heading(label,bookmark,eyebrow=None):
    items=[NextPageTemplate('opener'),PageBreak(),NextPageTemplate('body')]
    if eyebrow:items.append(Paragraph(html.escape(eyebrow.upper()),STYLES['eyebrow']))
    else:items.append(Spacer(1,35))
    p=Paragraph(html.escape(label),STYLES['chapter'])
    p.toc_label=f'{eyebrow.replace("Chapter ", "")} · {label}' if eyebrow and eyebrow.startswith('Chapter ') else label
    p.bookmark=bookmark
    items.append(p)
    return items


def paragraphs(body,note_map,prefix,used):
    blocks=re.split(r'\n\s*\n',body.strip());result=[];first=True
    for block in blocks:
        block=block.strip()
        if not block:continue
        if block.startswith('# '):continue
        if block.startswith('## ') or block.startswith('### '):
            result.append(Paragraph(xml(re.sub(r'^#{2,3}\s+','',block)),STYLES['section']))
            first=True;continue
        if block=='---':continue
        text=' '.join(block.splitlines())
        if re.match(r'^[-*]\s',text):text='• '+text[2:]
        result.append(Paragraph(xml(text,note_map,prefix,used),STYLES['first' if first else 'body']))
        first=False
    return result


def build_interior(plan,path):
    flow=[Spacer(1,90),Paragraph('SHADOW<br/>FUTURES',STYLES['title']),
          Paragraph(SUBTITLE,STYLES['subtitle']),Spacer(1,85),
          Paragraph('MARTIN ERLIC',STYLES['author']),PageBreak(),Spacer(1,220),
          Paragraph(f'Copyright © 2026 {AUTHOR}. All rights reserved.',STYLES['small']),
          Paragraph('First edition · October 2026',STYLES['small']),
          PageBreak(),Paragraph('Contents',STYLES['chapter'])]
    toc=TableOfContents();toc.levelStyles=[STYLES['toc']];flow.append(toc)
    flow.append(NextPageTemplate('body'))
    all_notes=[];counter=0;used=set();contents=[]
    for index,item in enumerate(plan['components']):
        text=(ROOT/item['file']).read_text(encoding='utf-8-sig')
        body,_,notes=text.partition('\n## Notes\n')
        definitions=re.findall(r'^\[\^([^\]]+)\]:\s*(.*?)(?=\n\[\^|\Z)',notes,re.M|re.S)
        definitions.sort(key=lambda pair: body.index(f'[^{pair[0]}]'))
        note_map={}
        for key,note_text in definitions:
            counter+=1;note_map[key]=counter;all_notes.append((item['title'],item['id'],key,counter,note_text.strip()))
        label=item['title'];chapter=index-1 if index>=2 and index<=13 else None
        eyebrow=f'Chapter {chapter}' if chapter else ('Prologue' if index==0 else 'Introduction' if index==1 else 'Epilogue')
        flow.extend(heading(label,item['id'],eyebrow))
        # The chapter title already appears in the opener.
        body=re.sub(r'^## '+re.escape(label)+r'\s*$', '',body,count=1,flags=re.M)
        flow.extend(paragraphs(body,note_map,item['id'],used))
        contents.append({'label':label,'section':eyebrow})
    flow.extend(heading('Notes','endnotes','Sources and further reading'))
    previous=None
    for chapter,prefix,key,number,text in all_notes:
        if chapter!=previous:
            flow.append(Paragraph(html.escape(chapter),STYLES['section']));previous=chapter
        dest=f'{prefix}-{key}'
        label=f'<a name="note-{dest}"/><link href="#ref-{dest}" color="#285f73"><b>{number}.</b></link> '
        flow.append(Paragraph(label+xml(' '.join(text.splitlines())),STYLES['note']))
    for item in plan['back_matter']:
        text=(ROOT/item['file']).read_text(encoding='utf-8-sig')
        label='References' if item['id']=='references' else 'Acknowledgments'
        flow.extend(heading(label,item['id'],'Back matter'))
        for block in re.split(r'\n\s*\n',text.strip()):
            if block.startswith('#'):continue
            flow.append(Paragraph(xml(' '.join(block.splitlines())),STYLES['note' if label=='References' else 'first']))
    doc=BookDoc(path);doc.multiBuild(flow,maxPasses=4)
    for item,entry in zip(plan['components'],contents):
        entry['page']=doc.section_pages[item['id']]+1
    return contents,counter


def raster_pdf(path,width,height,destination):
    document=pdfium.PdfDocument(str(path))
    page=document[0];image=page.render(scale=(width-0.001)/page.get_width()).to_pil()
    assert image.size==(width,height),(image.size,width,height)
    image.save(destination,quality=94,dpi=(300,300))
    document.close()


def build():
    for d in [OUT,PUBLIC,TEMP]:d.mkdir(parents=True,exist_ok=True)
    register_fonts();styles()
    plan=json.loads((ROOT/'book/planning/manuscript-plan.json').read_text(encoding='utf-8'))
    cover=PUBLIC/'shadow-futures-cover.png'
    if not cover.exists():raise FileNotFoundError('Copy the generated cover master into public/book first.')
    image=Image.open(cover);native=list(image.size)
    interior=OUT/'shadow-futures-interior-6x9.pdf'
    contents,note_count=build_interior(plan,interior)
    interior_pages=len(PdfReader(interior).pages)
    if interior_pages%2:
        writer=PdfWriter();writer.clone_document_from_reader(PdfReader(interior));writer.add_blank_page(*PAGE)
        with interior.open('wb') as f:writer.write(f)
        interior_pages+=1
    front=OUT/'shadow-futures-front-cover.pdf'
    c=canvas.Canvas(str(front),pagesize=PAGE);c.setTitle(TITLE+' — front cover');c.setAuthor(AUTHOR)
    c.drawImage(str(cover),0,0,*PAGE);c.showPage();c.save()
    # Reading copy includes the cover; print interior deliberately excludes it.
    reading=OUT/'shadow-futures.pdf';writer=PdfWriter()
    writer.clone_document_from_reader(PdfReader(interior));writer.insert_page(PdfReader(front).pages[0],index=0)
    writer.add_metadata({'/Title':f'{TITLE}: {SUBTITLE}','/Author':AUTHOR,
                         '/Subject':'Complete free reading edition · October 2026'})
    with reading.open('wb') as f:writer.write(f)
    shutil.copy2(reading,PUBLIC/reading.name)
    # Kindle's 1.6:1 canvas preserves the full 2:3 design without cropping text.
    ebook=TEMP/'kindle-canvas.pdf';w,h=384,614.4
    c=canvas.Canvas(str(ebook),pagesize=(w,h));c.setFillColor(colors.HexColor('#08212a'));c.rect(0,0,w,h,fill=1,stroke=0)
    c.drawImage(str(cover),0,(h-576)/2,w,576);c.showPage();c.save()
    raster_pdf(ebook,1600,2560,PUBLIC/'shadow-futures-kindle-cover.jpg')
    raster_pdf(front,1800,2700,OUT/'shadow-futures-print-front-300dpi.jpg')
    # KDP black-ink / white-paper full-wrap dimensions, with outer bleed.
    spine=interior_pages*0.002252*inch;bleed=0.125*inch
    width=2*PAGE[0]+spine+2*bleed;height=PAGE[1]+2*bleed
    wrap=OUT/'shadow-futures-kdp-paperback-cover.pdf'
    c=canvas.Canvas(str(wrap),pagesize=(width,height));c.setTitle(TITLE+' — 6×9 paperback cover');c.setAuthor(AUTHOR)
    c.setFillColor(INK);c.rect(0,0,width,height,fill=1,stroke=0)
    c.drawImage(str(cover),bleed+PAGE[0]+spine,0,PAGE[0]+bleed,height)
    backstyle=ParagraphStyle('back',fontName='Book',fontSize=12,leading=18,textColor=CREAM)
    text=[
        'A market can record every win and still lose sight of the alternatives.',
        'The same song finishes first in one market and fortieth in another. An early contract gives one firm the references its rival needs. A thousand AI agents produce answers—but who gets to test a different approach?',
        'Martin Erlic introduces the comparison budget: a way to examine the opportunities a reinforced market preserves for learning about alternatives. Following experiments, economic history and the emerging AI economy, Shadow Futures asks what success can tell us about contribution—and what it can leave unresolved.',
        'From work and capital ownership to competition and taxation, the book asks how institutions can preserve worthwhile discovery while keeping the benefits of scale.',
    ]
    y=height-63
    for i,t in enumerate(text):
        style=backstyle if i else ParagraphStyle('hook',parent=backstyle,fontSize=20,leading=25)
        p=Paragraph(html.escape(t),style);_,ph=p.wrap(PAGE[0]-96,height);p.drawOn(c,bleed+48,y-ph);y-=ph+22
    c.saveState();c.translate(bleed+PAGE[0]+spine/2,height/2);c.rotate(-90)
    c.setFillColor(CREAM);c.setFont('Book-Bold',15);c.drawCentredString(-20,-5,'SHADOW FUTURES')
    c.setFont('Sans',9);c.drawRightString(height/2-43,-4,'MARTIN ERLIC');c.restoreState()
    # Reserve a clear back-cover area for KDP's own barcode; do not invent an ISBN.
    c.setFillColor(CREAM);c.rect(bleed+PAGE[0]-171,bleed+24,144,86.4,fill=1,stroke=0)
    c.showPage();c.save()
    inputs=[{'file':i['file'],'sha256':sha(ROOT/i['file'])} for i in plan['components']+plan['back_matter']]
    record={'title':TITLE,'subtitle':SUBTITLE,'author':AUTHOR,'edition':'Free reading edition',
            'date':DATE,'pdf':'/book/shadow-futures.pdf','cover':'/book/shadow-futures-cover.png',
            'pages':len(PdfReader(reading).pages),'interior_pages':interior_pages,
            'pdf_bytes':reading.stat().st_size,'notes':note_count,'contents':contents,
            'trim_inches':[6,9],'kindle_cover_pixels':[1600,2560],
            'paperback':{'paper':'black ink / white paper','spine_inches':spine/inch,
                         'wrap_inches':[width/inch,height/inch],'bleed_inches':0.125},
            'cover_native_pixels':native,
            'production_note':'KDP-size exports are provided. Generated artwork is natively 1024×1536; 300-dpi raster exports are resampled, not added native detail. Check a print proof; higher-resolution original art is recommended before commercial print upload.',
            'manuscript_sha256':sha(ROOT/'book/drafts/Shadow_Futures_Complete_Manuscript.md'),
            'inputs':inputs,'pdf_sha256':sha(reading),
            'kdp_sources':['https://kdp.amazon.com/en_US/help/topic/G201953020',
                           'https://kdp.amazon.com/en_US/help/topic/G6GTK3T3NUHKLEFX',
                           'https://kdp.amazon.com/en_US/help/topic/GVBQ3CMEQW3W2VL6']}
    (ROOT/'book/research/publication-manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    public_keys=['title','subtitle','author','edition','date','pdf','cover','pages','pdf_bytes','contents']
    (PUBLIC/'edition.json').write_text(json.dumps({k:record[k] for k in public_keys},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['pages','interior_pages','notes','pdf_bytes','paperback','pdf_sha256']},indent=2))


if __name__=='__main__':build()
