BiLSTM_CRF实体识别
fasttext预测物品分类
gensim案件相似度计算


一、环境
Python 3.6.8

依赖及安装(见requirements.txt)
fasttext==0.9.2
gensim==3.8.3
numpy==1.16.4
tensorflow==1.2.0


pip install -r requirements.txt


二、说明

1、data目录为预料目录，save为保存模型目录。


2、语料

data/ner_corpus.txt，BiLSTM_CRF原始语料（案件信息）。
如下：
2019年11月28日16时39分许，被害人郭万翔拨打110报警称：2019年11月28日16时30分许，在辽宁省大连市甘井子区山东路118号楼楼下，其发现自己停放在此处的白色本田轿车，车牌号码为辽B3F0Z6的主驾驶室玻璃被砸碎，车内一个棕色小挎包及一个黑色钱包被盗，包内有300元现金及一些银行卡，总损失价值1000元左右。

data/ner_train.txt，BiLSTM_CRF标注语料用于训练模型。


data/classifier_corpus.txt，物品原始语料。
data/classifier_train.txt，物品训练语料，需要跟进classifier_corpus.txt手动处理，用于fasttext预测物品名称分类。
如下：
__label__首饰 , 钻戒
__label__银行卡 , 储蓄卡

save/ner_result.txt，基于ner_corpus.txt预料以BiLSTM_CRF实体识别后的结果，用于gensim计算相似度。
以tab分割，前半部分是实体(日期、地址、物品名称已经处理)。后半部分为原始语料。
如下：
下午 星期四 门址 车牌号码为辽B3 驾驶室玻璃 网络	2019年11月28日16时39分许，被害人郭万翔拨打110报警称：2019年11月28日16时30分许，在辽宁省大连市甘井子区山东路118号楼楼下，其发现自己停放在此处的白色本田轿车，车牌号码为辽B3F0Z6的主驾驶室玻璃被砸碎，车内一个棕色小挎包及一个黑色钱包被盗，包内有300元现金及一些银行卡，总损失价值1000元左右。

3、步骤
第一步：生成词典
基于ner_train.txt语料生成词典，文件位于save/word2id.pkl。
注：重复执行自动覆盖。

python3 main.py --mode=ner_gendict

第二步：训练模型
基于ner_train.txt训练BiLSTM_CRF模型。
注：重复生成新的模型目录,如save/1603464530。

python3 main.py --mode=ner_train

第三步：实体识别
根据第二步生成的模型处理案件语料生成gensim语料。

注：--model参数不许指定模型目录。

python3 main.py --mode=tags --model=1603464530


第四步：预测

注：--model参数不许指定模型目录。

python3 main.py --mode=gensim_predict --model=1603464530

其他：实体识别演示
注：--model参数不许指定模型目录。

python3 main.py --mode=ner_demo --model=1603464530

