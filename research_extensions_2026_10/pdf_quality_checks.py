"""Render exports/poster and check text bounds; originals remain untouched."""
import json
from pathlib import Path
import fitz
from PIL import Image,ImageOps,ImageDraw
from research_extensions_2026_10.run_study import OUT,RESULTS,write_json


def main():
    qa=OUT/'presentation/qa';qa.mkdir(parents=True,exist_ok=True)
    checks=[];previews=[]
    paths=sorted((RESULTS/'live').glob('*.pdf'))+sorted((OUT/'presentation').glob('*.pdf'))
    for path in paths:
        with fitz.open(path) as doc:
            overflow=[]
            for i,page in enumerate(doc):
                for block in page.get_text('blocks'):
                    if block[6]!=0:continue
                    rect=fitz.Rect(block[:4])
                    if rect.x0< -2 or rect.y0< -2 or rect.x1>page.rect.width+2 or rect.y1>page.rect.height+2:
                        overflow.append({'page':i+1,'bounds':list(rect),'text':block[4][:80]})
            outputs=[]
            for page_index,page in enumerate(doc):
                output=qa/(path.stem+(f'_page{page_index+1}' if page_index else '')+'.png')
                scale=min(2.,2400/page.rect.width)
                page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(output)
                previews.append(output);outputs.append(output.name)
            checks.append({'file':path.name,'pages':len(doc),'text_outside_page':overflow,'rendered_previews':outputs})
    thumbs=[]
    for path in previews:
        with Image.open(path) as im:
            im=im.convert('RGB');im.thumbnail((570,800));tile=Image.new('RGB',(600,840),'#e7e7e7');tile.paste(im,((600-im.width)//2,30))
            ImageDraw.Draw(tile).text((12,10),path.stem,fill='black');thumbs.append(tile)
    for group_start in range(0,len(thumbs),9):
        group=thumbs[group_start:group_start+9]
        sheet=Image.new('RGB',(1800,840*((len(group)+2)//3)),'white')
        for i,tile in enumerate(group):sheet.paste(tile,((i%3)*600,(i//3)*840))
        sheet.save(qa/('contact_sheet'+(f'_{group_start//9+1}' if group_start else '')+'.png'))
    write_json(RESULTS/'pdf_quality_checks.json',{'files':checks,'all_pages_within_bounds':all(not c['text_outside_page'] for c in checks),
        'visual_review':'Rendered previews/contact sheet require inspection; structural bounds alone do not prove good layout.'})
    print('Rendered',len(checks),'PDF artifacts for visual review.',flush=True)


if __name__=='__main__':main()
