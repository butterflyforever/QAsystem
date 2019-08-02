
def get_url(host, port):
    return 'http://' + host + ':' + port

urls = dict()

rule_trigger_prob = 0.8
rule_host = '0.0.0.0'
rule_port = '5000'
urls['rule'] = get_url(rule_host, rule_port)

retrieval_host = '0.0.0.0'
retrieval_port = '8888'
retrieval_threshold = 0.8
retrieval_soft_threshold = 0.5
urls['retrieval'] = get_url(retrieval_host, retrieval_port)

seq2seq_host = '0.0.0.0'
seq2seq_port = '5001'
urls['seq2seq'] = get_url(seq2seq_host, seq2seq_port)

dpgan_host = '0.0.0.0'
dpgan_port = '8890'
urls['dpgan'] = get_url(dpgan_host, dpgan_port)

gif_csv = '/data/zhc/data/gif/gif.csv'
gif_trigger_prob = 0
gif_sim_thres = 0.1
gif_k = 5
gif_host = '0.0.0.0'
gif_port = '8891'
urls['gif'] = get_url(gif_host, gif_port)
