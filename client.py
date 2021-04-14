import requests
import argparse
import itertools
import time

parser = argparse.ArgumentParser(description='Flask server script')
parser.add_argument('--port', type=int, default=8081, help='port number for api')
args = parser.parse_args()
s = requests.session()
ip_address = 'http://0.0.0.0:' + str(args.port)

with open('Dialogflow_Response.txt', encoding='utf-8') as f:
    origin_sentences = [line.strip() for line in f]
    
input_list = list()
for line in origin_sentences:
    if len(line.split('#')) > 1:
        input_list.append((line.strip().split('#')[0].strip(), line.strip().split('#')[1][:5]))

aa = time.time()
for cur in input_list:
    r = s.post(ip_address, data={'sentence': cur[0], 'emotion': '{}'.format(str(cur[1])), 'intensity': 1})
print('It takes {}s'.format(time.time() - aa))