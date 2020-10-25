import fasttext

def train(): # 训练模型
    # model = fasttext.train_supervised("./classifier_train.txt", lr=0.1, dim=20, epoch=100, wordNgrams=3, loss='softmax')
    model = fasttext.train_supervised("./data_path/classifier_train.txt", lr=1.0, dim=50, epoch=25, wordNgrams=1, minCount=1, loss='softmax')
    # print(model.words)
    model.save_model("./data_path/classifier_model")

def test(): # 预测
    classifier = fasttext.load_model("./data_path/classifier_model")
    #result = classifier.test("classifier_test.txt")
    #print("准确率:", result)
    testing=['盗刷']
    print(testing, classifier.predict(testing, k=7))

if __name__ == '__main__':
    train()
    test()