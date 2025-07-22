# Run Order .sql.back to final csv

1) run ts_db_script.sh                          (postgresql db > hv_ts.csv)
3) go to Colab run app.py                       (hv_ts.csv > df_pivoted.csv)
4) run app_nan.py                               (df_pivoted.csv > df_cleaned.csv)
5) run preprocessing.py                         (df_cleaned.csv > df_ml_ready.csv)