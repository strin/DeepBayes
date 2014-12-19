for num_node in 50 200 500
do
  for kappa in 0 0.1
  do
    for sigma in 0 0.01 0.1
    do
      nohup python test_mnist.py $num_node $kappa $sigma &
    done
  done
done
