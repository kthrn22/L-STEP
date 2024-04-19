# TempGT

All datasets can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o)

Install linforer from [linformer](https://github.com/lucidrains/linformer) or 
```
pip install linformer
```

Train TempGT (uci dataset as example)
```
--dataset uci --model_name TempGT --batch_size 200 --num_epochs 10 --time_gap 2000 --num_neighbors 30 --pe_weight 0.3 --num_heads 4 --transformer_depth 12 --gpu 3
```