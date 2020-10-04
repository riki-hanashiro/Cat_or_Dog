#  coding: utf-8

import numpy as np
from simple_convnet import SimpleConvNet
from trainer import Trainer
from load_CorD import *

load_num=8000
train_data_num=7000
test_data_num=2000

#(x_train, t_train), (x_test, t_test)=load_data_list(image_size,load_num,train_data_num,test_data_num)

#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]
max_epochs = 20
acc_dict={}
tmp_acc_list=[]
for size in range(22,23,2):
    tmp_acc_list.clear()
    print("size ---> ",size)
    print("Now calculating...... ")
    for j in range(1):
        input_data_dim=(3, size, size)
        (x_train, t_train), (x_test, t_test)=load_data_list( size, load_num, train_data_num, test_data_num)
        network = SimpleConvNet(PRE_DATA=True, input_dim=input_data_dim,
                    conv_param={"filter_num": 30,"filter_size": 5,"pad": 0,"stride": 1},
                    hidden_size=100, output_size=2, weight_init_std=0.01)

        trainer = Trainer(network, x_train, t_train, x_test, t_test,
            epochs=max_epochs, mini_batch_size=100,
            optimizer="Adam", optimizer_param={"lr":0.001},
            evaluate_sample_num_per_epoch=1000)
        trainer.train( size, tmp_acc_list)
    #acc_dict[size]=sum(tmp_acc_list)/10
    #print(tmp_acc_list)
    #print(acc_dict)

"""
with open ( "each_acc.txt", mode = "w") as f:
    for key,value in acc_dict.items():
        f.write(f'{key} {value}\n')
"""

#パラメータの保存
network.save_params("params_color.pkl")
print("Saved Network Parameters!")
"""
#グラフの描画
markers = {"train":"o", "test":"s"}
x=np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
"""
