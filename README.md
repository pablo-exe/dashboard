# Dashboard (Streamlit)

## Ejecutar en servidor

1) Instala dependencias:
```
pip install -r requirements.txt
```

2) (Opcional) Define la ruta a la base DuckDB:
```
setx EXPERIMENTS_DB_PATH "C:\ruta\experiments.duckdb"
```

3) Lanza el dashboard:
```
streamlit run app.py
```

Si no se define `EXPERIMENTS_DB_PATH`, el dashboard usa:
`dashboard/data/experiments.duckdb`
