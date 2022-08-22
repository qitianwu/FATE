models=('partial' 'avg' 'knn' 'local')
datasets=('calls' 'gene' 'protein' 'github' 'drive' 'robot')

for d in ${datasets[@]}
do
  for m in ${models[@]}
  do
    python main.py --dataset $d --model $m --feature_ratio 0.8 --new_feature_ratio 0.2
    python main.py --dataset $d --model $m --feature_ratio 0.7 --new_feature_ratio 0.3
    python main.py --dataset $d --model $m --feature_ratio 0.6 --new_feature_ratio 0.4
    python main.py --dataset $d --model $m --feature_ratio 0.5 --new_feature_ratio 0.5
    python main.py --dataset $d --model $m --feature_ratio 0.4 --new_feature_ratio 0.6
    python main.py --dataset $d --model $m --feature_ratio 0.3 --new_feature_ratio 0.7
  done
done