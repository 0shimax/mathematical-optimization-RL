# easy
python ddpg_mo.py --env easy2d --steps 100000 --eval-interval 10000 --obs-dim 2 --action-low -100 --action-high 100 --reward-scale-factor 0.02 --sigma-scale-factor 0.05 --n-hidden-layers 10 --n-hidden-channels 128

# Griewank function
python ddpg_mo.py --env griewank --steps 100000 --eval-interval 10000 --obs-dim 101 --action-dim 100 --action-low -600 --action-high 600 --reward-scale-factor 0.02 --sigma-scale-factor 0.05 --n-hidden-layers 10 --n-hidden-channels 128
