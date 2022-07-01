import config
os = config.os

# os.system('python ./gen_i_o.py') # 生成数据

model_name = "cnn1d"

os.system('python ./main_lstm.py') 
os.system('python ./plot_all.py') 

