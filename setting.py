m_sent_len = 50
m_sent_len_o = 5000
m_word_dim = 50
m_batch_size = 50
m_text_size = 500

#m_path = "data/network_diagnosis_data/cut1000/"
m_path = "/media/zui/work/NETWORK/network_diagnosis_data/"
m_model_w2v_name = "word_emb_size50wind100count100iter30"
m_epoch = 5

m_learning_rate = 0.1

m_labels = ["NORMAL",
            "GTPC_TUNNEL_PATH_BROKEN",
            "Paging",
            "UeAbnormal"]

m_names = ["BaseLine-BigData_1kUE_20ENB_NORMAL-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_gtpcbreakdown-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_paging-Case_Group_1-Case_1",
           "BaseLine-BigData_1kUE_20ENB_UeAbnormal-Case_Group_1-Case_1_new_With_Tag"]


m_dic_size = 100  # 99+1
m_hidden_dim = 72
"""
imdb_sent_len = 5000
imdb_sent_len_o = 1000
imdb_word_dim = 50
imdb_text_size = 100
imdb_batch_size = 100
imdb_batch_size_o = 100
imdb_path = "/media/zui/work/NETWORK/aclImdb/"
imdb_model_w2v_name = "imdb_model_count100size50"
imdb_learning_rate = 0.1
"""