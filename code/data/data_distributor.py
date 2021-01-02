from collections import defaultdict
import logging
import numpy as np
import csv
import json
import os
import random

logger = logging.getLogger(__name__)


class DataDistributor:
    def __init__(self, triple_per_agent_limit = None):
        self.triple_per_agent = {}
        self.agent_entity_vocab = {}
        self.agent_relation_vocab = {}
        self.triple_per_agent_limit = triple_per_agent_limit

    def split(self, params, agent_names):
        if params['split_random']:
            self.split_grapher_triple_random(params, agent_names)
        else:
            self.split_grapher_triple(params, agent_names)
        self.split_batcher_triple(params, agent_names)
        # self.create_vocab(params, agent_names)

    def set_triple_per_agent_limit(self, triple_per_agent_limit):
        self.triple_per_agent_limit = triple_per_agent_limit

    def get_grapher_triple_per_count(self):
        return self.triple_per_agent

    def get_grapher_entity_per_count(self):
        return self.agent_entity_vocab

    def get_grapher_relation_per_count(self):
        return self.agent_relation_vocab

    def split_grapher_triple(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'graph.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            triple_count_start_idx = 0
            if self.triple_per_agent_limit:
                triple_count_per_agent = self.triple_per_agent_limit
            else:
                triple_count_per_agent = int(len(triple_file) / len(agent_names))
            self.triple_per_agent = {}
            self.agent_entity_vocab = {}
            self.agent_relation_vocab = {}

            for agent in agent_names:
                self.triple_per_agent[agent] = triple_count_per_agent
                self.agent_entity_vocab[agent] = []
                self.agent_relation_vocab[agent] = []
                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(triple_count_start_idx, triple_count_start_idx + triple_count_per_agent):

                        if not triple_file[i][0] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[i][0])
                        if not triple_file[i][1] in self.agent_relation_vocab[agent]:
                            self.agent_relation_vocab[agent].append(triple_file[i][1])
                        if not triple_file[i][2] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[i][2])

                        writer.writerow(triple_file[i])
                    triple_count_start_idx = triple_count_start_idx + triple_count_per_agent

    def split_grapher_triple_random(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'graph.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            self.triple_per_agent = {}
            self.agent_entity_vocab = {}
            self.agent_relation_vocab = {}

            agent_triple_spilt_param = {}

            agent_triple_spilt_param[agent_names[0]] = 0.10
            agent_triple_spilt_param[agent_names[1]] = 0.30
            agent_triple_spilt_param[agent_names[2]] = 0.70

            for agent in agent_names:
                triple_count_per_agent = int(len(triple_file) * agent_triple_spilt_param[agent])
                self.triple_per_agent[agent] = triple_count_per_agent
                self.agent_entity_vocab[agent] = []
                self.agent_relation_vocab[agent] = []

                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(triple_count_per_agent):
                        idx = random.randint(1, len(triple_file) - 1)

                        if not triple_file[idx][0] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[idx][0])
                        if not triple_file[idx][1] in self.agent_relation_vocab[agent]:
                            self.agent_relation_vocab[agent].append(triple_file[idx][1])
                        if not triple_file[idx][2] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[idx][2])

                        writer.writerow(triple_file[idx])
                        triple_file.remove(triple_file[idx])

    def split_batcher_triple(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'train.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            triple_count_start_idx = 0
            triple_count_per_agent = int(len(triple_file) / len(agent_names))

            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'train_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(triple_count_start_idx, triple_count_start_idx + triple_count_per_agent):
                        writer.writerow(triple_file[i])
                    triple_count_start_idx = triple_count_start_idx + triple_count_per_agent

        with open(params['data_input_dir'] + '/' + 'dev.txt') as triple_file_raw:
            dev_triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            dev_triple_count_start_idx = 0
            dev_triple_count_per_agent = int(len(dev_triple_file) / len(agent_names))
            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'dev_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(dev_triple_count_start_idx, dev_triple_count_start_idx + dev_triple_count_per_agent):
                        writer.writerow(dev_triple_file[i])
                    dev_triple_count_start_idx = dev_triple_count_start_idx + dev_triple_count_per_agent


    # def split_batcher_triple(self, params, agent_names):
    #     for agent in agent_names:
    #         with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt') as triple_file_raw:
    #             triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
    #             batcher_idx = np.random.randint(len(triple_file), size=(1, 10))
    #             with open(params['data_input_dir'] + '/' + 'train_' + agent + '.txt', 'w') as triple_file_name:
    #                 writer = csv.writer(triple_file_name, delimiter='\t')
    #                 for i in range(0, len(batcher_idx[0])):
    #                     writer.writerow(triple_file[batcher_idx[0][i]])
    #
    #     with open(params['data_input_dir'] + '/' + 'dev.txt') as triple_file_raw:
    #         dev_triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
    #
    #         dev_triple_count_start_idx = 0
    #         dev_triple_count_per_agent = int(len(dev_triple_file) / len(agent_names))
    #         for agent in agent_names:
    #             with open(params['data_input_dir'] + '/' + 'dev_' + agent + '.txt', 'w') as triple_file_name:
    #                 writer = csv.writer(triple_file_name, delimiter='\t')
    #                 for i in range(dev_triple_count_start_idx, dev_triple_count_start_idx + dev_triple_count_per_agent):
    #                     writer.writerow(dev_triple_file[i])
    #                 dev_triple_count_start_idx = dev_triple_count_start_idx + dev_triple_count_per_agent


    def create_vocab(self, params, agent_names):
        params['relation_vocab'] = json.load(open(params['vocab_dir'] + '/relation_vocab.json'))
        params['entity_vocab'] = json.load(open(params['vocab_dir'] + '/entity_vocab.json'))

        print('Total Entity')
        print(len(params['entity_vocab']))
        print('Total Relation')
        print(len(params['relation_vocab']))

        for agent in agent_names:
            with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt') as triple_file_raw:
                triple_file = csv.reader(triple_file_raw, delimiter='\t')

                agent_entity_vocab = {}
                agent_relation_vocab = {}
                for line in triple_file:
                    if not line[0] in agent_entity_vocab.keys():
                        agent_entity_vocab[line[0]] = params['entity_vocab'][line[0]]
                    if not line[1] in agent_relation_vocab.keys():
                        agent_relation_vocab[line[1]] = params['relation_vocab'][line[1]]
                    if not line[2] in agent_entity_vocab.keys():
                        agent_entity_vocab[line[2]] = params['entity_vocab'][line[2]]

                print(agent + ' Entity')
                print(len(agent_entity_vocab))
                print(agent + ' Relation')
                print(len(agent_relation_vocab))

                if not os.path.exists(params['vocab_dir'] + '/' + agent):
                    os.makedirs(params['vocab_dir'] + '/' + agent)

                with open(params['vocab_dir'] + '/' + agent + '/entity_vocab.json', 'w') as jsonWriter:
                    json.dump(agent_entity_vocab, jsonWriter)
                with open(params['vocab_dir'] + '/' + agent + '/relation_vocab.json', 'w') as jsonWriter:
                    json.dump(agent_relation_vocab, jsonWriter)