# DCN+AFM

- main_5158a232.py
    - 5158a232-dcan-120-6.csv: 6.063812
    - 5158a232-dcan-118-4.csv: 6.070348
- main_393efecd.py
    - 393efecd-dcan-2015-9.csv: 6.068042
- main_56464278.py
    - 56464278-dcan-84-9.csv: 6.068146
- main_b703070a.py
    - b703070a-dcan-410-10.csv: 6.071503
    - b703070a-dcan-403-13.csv: 6.072262
    - b703070a-dcan-403-11.csv: 6.072530

- ensemble above 7 files: 6.026862

```bash
python ensemble.py 5158a232-dcan-120-6.csv,393efecd-dcan-2015-9.csv,56464278-dcan-84-9.csv,5158a232-dcan-118-4.csv,b703070a-dcan-410-10.csv,b703070a-dcan-403-13.csv,b703070a-dcan-403-11.csv -m linear -o dcan.csv --id_name row_id --target_name is_installed
```