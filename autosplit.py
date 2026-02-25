from ultralytics.data.split import autosplit

autosplit(
    path='Hammer/train/images', 
    weights=(0.8, 0.2, 0.0), # 80% train, 20% val
    annotated_only=False
)