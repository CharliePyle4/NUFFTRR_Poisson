import json
import pandas as pd

with open('Tests/CPU/testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/tables_output.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and cell.get('outputs'):
            for out in cell['outputs']:
                if 'data' in out and 'text/html' in out['data']:
                    html = "".join(out['data']['text/html'])
                    try:
                        dfs = pd.read_html(html)
                        for df in dfs:
                            out_f.write(f"=== Cell {i} Table ===\n")
                            out_f.write(df.to_string() + "\n\n")
                    except Exception as e:
                        out_f.write(f"Error parsing html in cell {i}: {e}\n")
                elif 'text' in out:
                    txt = "".join(out['text'])
                    for l in txt.split('\n'):
                        if any(k in l for k in ["Accuracy", "Table", "Problem", "===", "label", "L2", "rel"]):
                            out_f.write(f"--- Cell {i} Text: {l}\n")
