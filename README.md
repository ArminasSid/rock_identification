# rock_identification
Atpažįstam akmenis

Failai pateikiami:
Data/Vadagiai-0.967-210713
Data/Jura-1.98-210916
....

# Konvertuojam visus .shp failus į .geojson failus. Pridedam offset and geojson.
```python convertshapefile.py```

# Pagaminam dataset-generation, kurie bus naudojami duomenu rinkiniu generavimui
```python warp_to_cutline.py```

# Kuriam duomenu rinkini
```python generatedataset_k_fold_v2.py```

# Konvertuoti dataset i yolov5 tipa su roboflow.com

# Treniruojam duomenų rinkinį
```python train.py```

# Aptažįstam vaizdus
```python predict.py```
