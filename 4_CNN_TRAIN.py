import numpy as np        
import cnn_sgn

IMG_SIZE = 96
LR = 1e-3  #.001 learing rate
nb_classes=27
MODEL_NAME = 'handsign.model'

def one_hot_targets_(Y,nb_classes):
    targets = np.array(Y).reshape(-1)
    print(targets)
    one_hot_targets_ = np.eye(nb_classes)[targets]
    return one_hot_targets_
train_data = np.load('train_data.npy',encoding="latin1",allow_pickle=True)

train = train_data[:]
test = train_data[:7000]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
Y1=one_hot_targets_(Y,nb_classes)
 
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
test_y1=one_hot_targets_(test_y,nb_classes)
test_y=test_y1
Y=Y1

model=cnn_sgn.cnn_model()
model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
score = model.evaluate(test_x, test_y)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
