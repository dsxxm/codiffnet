## 项目结构

项目根目录/
├── conf //项目配置文件
│   ├── flickr_diffnetplus.ini
│   └── yelp_diffnetplus.ini
├── data //数据集
│   ├── flickr
│   │   ├── flickr.links
│   │   ├── flickr.rating
│   │   ├── flickr.test.rating
│   │   ├── flickr.train.rating
│   │   ├── flickr.val.rating
│   │   ├── item_vector.npy
│   │   └── user_vector.npy
│   └── yelp
│       ├── item_vector.npy
│       ├── user_vector.npy
│       ├── yelp.links
│       ├── yelp.test.rating
│       ├── yelp.train.rating
│       └── yelp.val.rating
├── diffnetplus.py //模型核心代码
├── entry.py //程序入口
├── log //日志文件
├── Readme.md //Readme
├── requirements.txt //依赖
├── train.py //训练程序
└── utils //工具类 包含读取数据集 数据集预处理 打印日志 读取配置
    ├── DataModule.py
    ├── DataUtil.py
    ├── Evaluate.py
    ├── __init__.py
    ├── Logging.py
    └── ParserConf.py

## 项目依赖

python 3.7 tensorflow1.15 更多配置可在requirements.txt中查看

## 数据集

数据集在[Datesets](https://drive.google.com/drive/folders/1YAJvgsCJLKDFPVFMX3OG7v3m1LAYZD5R)中获取

## 如何运行

python entry.py --dataname=\<dataname\> --model_name=\<model_name\> --gpu=\<gpuid\>
e.g.

- python entry.py --dataname=yelp --model_name=diffnetplus --gpu=0
