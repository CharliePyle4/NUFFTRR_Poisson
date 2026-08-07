import json

with open('Tests/CPU/testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))[:120].replace('\n', ' ')
    print(f"Cell {i} [{cell['cell_type']}]: {src}")
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            if 'text' in out:
                txt = "".join(out['text'])
                lines = [line for line in txt.split('\n') if not line.startswith('Pipeline')]
                print("   Text summary: " + "\n".join(lines[:10]))
            if 'data' in out:
                if 'text/html' in out['data']:
                    html = "".join(out['data']['text/html'])
                    import re
                    # parse html table
                    import pandas as pd
                    try:
                        dfs = pd.read_html(html)
                        for df in dfs:
                            print("   HTML DataFrame:\n", df.to_string())
                    except Exception as e:
                        print("   HTML parse error:", e)
                elif 'text/plain' in out['data']:
                    print("   Plain:", "".join(out['data']['text/plain'])[:200])
