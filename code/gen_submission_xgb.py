import pandas as pd

df = pd.read_csv("predictions_xgb.csv")
#df2 = pd.read_csv("constant_predictions_xgb.csv")

#df = pd.concat([df1, df2], ignore_index=True)

df_train = pd.read_csv('../data/train_y_v0.1.0.csv')
out = pd.DataFrame(df['filename'].copy())

for col in df_train.columns[1:]:
    out[col] = 0.0
    
classes = set(df['predicted_class'])
for c in classes:
  filenames = df[df['predicted_class']==c]['filename']
  sensor_types = [s.strip() for s in c.split(', ')]
  mask = out['filename'].isin(filenames)
  out.loc[mask, sensor_types] = 1.0

out.to_csv("test_prediction_clean.csv.gz",compression="gzip")    