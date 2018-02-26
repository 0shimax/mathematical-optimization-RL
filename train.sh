# easy
python ddpg_mo.py --env easy2d --steps 1000000 --eval-interval 10000 --obs-dim 2 --action-low -100 --action-high 100 --reward-scale-factor 10

# Griewank function
python ddpg_mo.py --env griewank --steps 1000000 --eval-interval 10000 --obs-dim 201 --action-dim 200 --action-low -600 --action-high 600 --reward-scale-factor 10
