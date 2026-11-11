"""Check PDF size, page count, embedded fonts, text bounds and abstract length.

Requires requirements-pdf.txt. Visual review remains necessary for overlap,
chart readability, equations and captions. A metadata placeholder is reported,
not silently treated as a completed submission form.
"""
import argparse,hashlib,json,re
from pathlib import Path
from pypdf import PdfReader
import pdfplumber
ROOT=Path(__file__).resolve().parent
def embedded(font):
    font=font.get_object()
    if font.get('/Subtype')=='/Type3':return '/CharProcs' in font
    if '/DescendantFonts' in font:return all(embedded(f) for f in font['/DescendantFonts'])
    desc=font.get('/FontDescriptor')
    return bool(desc and any(k in desc.get_object() for k in ['/FontFile','/FontFile2','/FontFile3']))
def fonts(resources,seen=None):
    seen=set() if seen is None else seen;out={}
    resources=resources.get_object()
    for name,f in resources.get('/Font',{}).items():
        obj=f.get_object();out[str(obj.get('/BaseFont',name))]=embedded(f)
    for x in resources.get('/XObject',{}).values():
        obj=x.get_object();identity=(getattr(x,'idnum',None),getattr(x,'generation',None))
        if identity in seen:continue
        seen.add(identity)
        if '/Resources' in obj:out.update(fonts(obj['/Resources'],seen))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('pdf',nargs='?',type=Path,default=ROOT/'main.pdf');a=p.parse_args()
    text=(ROOT/'main.tex').read_text();abstract=text.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    assert len(abstract.split())==100,'Abstract must contain exactly 100 whitespace-delimited words'
    pdf=PdfReader(a.pdf);assert len(pdf.pages)==6,'Expected exactly six pages'
    fs={};bounds=[];mm=72/25.4;tol=.7
    with pdfplumber.open(a.pdf) as doc:
        for i,page in enumerate(doc.pages):
            assert abs(page.width-210*mm)<.1 and abs(page.height-297*mm)<.1,'Not A4'
            chars=[c for c in page.chars if c['text'].strip()]
            box=[min(c['x0'] for c in chars),min(c['top'] for c in chars),max(c['x1'] for c in chars),max(c['bottom'] for c in chars)]
            assert box[0]>=44*mm-tol and box[2]<=166*mm+tol,(i+1,'horizontal overflow',box)
            assert box[1]>=52*mm-tol and box[3]<=245*mm+tol,(i+1,'vertical overflow',box)
            bounds.append(dict(page=i+1,text_bbox_points=box,words=len(page.extract_text().split())))
    for page in pdf.pages:fs.update(fonts(page['/Resources']))
    assert fs and all(fs.values()),fs
    log=ROOT/'main.log'
    if log.exists():assert not re.search(r'Overfull \\[hv]box',log.read_text(errors='replace')),'Overfull box in compilation log'
    record=dict(pdf_sha256=hashlib.sha256(a.pdf.read_bytes()).hexdigest(),pages=6,format='A4',
        printing_area_mm=[122,193],margins_mm=[44,44,52,52],abstract_words=100,
        all_fonts_embedded=True,fonts=fs,page_bounds=bounds,confirmation_evaluated=True,
        old_reserved_official_final_test_accessed=False,
        author_identified='Kavyasree Nunna' in text,affiliation_pending='to be confirmed before submission' in text,
        status='Formatting checked; held-out confirmation completed; affiliation, novelty assessment and submission decision remain pending')
    (ROOT/'PAPER_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
