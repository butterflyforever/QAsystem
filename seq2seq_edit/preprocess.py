from config import config
import os
from collections import Counter


def load_weibo():
    module_path = os.path.dirname(__file__)
    data_path = "../raw_chat_corpus/weibo-400w"
    post_path = module_path + "/" + data_path + "/stc_weibo_train_post"
    response_path = module_path + "/" + data_path + "/stc_weibo_train_response"
    train_data_path = module_path + "/" + config["train_data"]
    voc_path = module_path + "/" + config["word_dict_path"]

    voc = Counter()
    count = 100000
    f1 = open(post_path, 'r', encoding="utf8")
    f2 = open(response_path, 'r', encoding="utf8")
    f3 = open(train_data_path, 'w', encoding="utf8")
    f4 = open(voc_path, 'w', encoding="utf8")
    for post in f1:
        response = f2.readline().strip("\r\n")
        post = post.strip("\r\n")
        # print(post.split())
        # print(post.split(""))
        f3.write(post+"\t"+response+"\n")
        for i in post.split():
            voc.update([i])
        for i in response.split():
            voc.update([i])
        count -= 1
        if count <= 0:
            break
    # print(voc)

    for item in voc.most_common()[:4000]:
        f4.write(item[0] + "\n")
    f4.write("<PAD>\n")
    f4.write("<UNK>\n")
    f4.write("<EOS>\n")
    f4.write("<GO>\n")

    f1.close()
    f2.close()
    f3.close()
    f4.close()




if __name__ == "__main__":
    load_weibo()