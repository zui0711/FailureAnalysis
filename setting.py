#from utils import *
from gensim.models import Word2Vec

m_dic_size = 100  # 99+1
m_hidden_dim = 72

m_sent_len = 20
m_word_dim = 50
m_batch_size = 10

#m_path = "data/network_diagnosis_data/cut1000/"
m_path = "data/network_diagnosis_data/"
m_model_w2v_name = "word_emb_size50wind100count100iter30"
m_epoch = 5

m_labels = ["NORMAL",
            "GTPC_TUNNEL_PATH_BROKEN",
            "Paging",
            "UeAbnormal"]

m_names = ["BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"]

m_learning_rate = 0.01
