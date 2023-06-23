import pickle
import os
import random
import json
import csv
import numpy as np
import torch
from Node import Node
from FusionModel import translator
from schemes import Scheme


def num2ord(num):
    if num % 10 == 1:
        ord_str = str(num) + 'st'
    elif num % 10 == 2:
        ord_str = str(num) + 'nd'
    elif num % 10 == 3:
        ord_str = str(num) + 'rd'
    else:
        ord_str = str(num) + 'th'
    return ord_str


class MCTS:
    def __init__(self, search_space, tree_height, arch_code_len):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space
        self.ARCH_CODE_LEN  = arch_code_len
        self.ROOT           = None
        self.Cp             = 0.5
        self.nodes          = []
        self.samples        = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list       = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(None, is_good_kid, self.ARCH_CODE_LEN, True))
            else:
                self.nodes.append(Node(self.nodes[parent_id], is_good_kid, self.ARCH_CODE_LEN, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.init_train()


    def init_train(self):
        for i in range(0, 200):
            net = random.choice(self.search_space)
            self.search_space.remove(net)
            self.TASK_QUEUE.append(net)
            self.sample_nodes.append('random')

        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")


    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)


    def populate_prediction_data(self):
        self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train()


    def predict_nodes(self):
        for i in self.nodes:
            i.predict()


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag)
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            curt_node = curt_node.kids[np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]]

        return curt_node


    def evaluate_jobs(self):
        while len(self.TASK_QUEUE) > 0:
            job = self.TASK_QUEUE.pop()
            sample_node = self.sample_nodes.pop()
            
            try:
                print("\nget job from QUEUE:", job)
                
                job_str = json.dumps(job)
                design = translator(job)
                print("translated to:\n{}".format(design))
                print("\nstart training:")
                best_model, report = Scheme(design)
                maeinv = 1 / report['metrics']['mae']
    
                self.DISPATCHED_JOB[job_str] = maeinv
                self.samples[job_str]        = maeinv
                self.mae_list.append(1 / maeinv)
                print("mae: {}".format(1/maeinv))
                with open('results.csv', 'a+', newline='') as res:
                    writer = csv.writer(res)
                    best_val_loss = report['best_val_loss']
                    metrics = report['metrics']
                    writer.writerow([len(self.samples), job_str, sample_node, best_val_loss, metrics['mae'], metrics['corr'],
                                     metrics['multi_acc'], metrics['bi_acc'], metrics['f1']])
                print("\nresults of current model saved")
                # save all models and reports
                torch.save(best_model.state_dict(), 'models/model_weights_'+str(len(self.samples))+'.pth')
                with open('reports/report_'+str(len(self.samples)), 'wb') as file:
                    pickle.dump(report, file)
                if maeinv > self.MAX_MAEINV:
                    self.MAX_MAEINV = maeinv
                    self.MAX_SAMPNUM = len(self.samples)
                    torch.save(best_model.state_dict(), 'model_weights.pth')
                    with open('report', 'wb') as file:
                        pickle.dump(report, file)
                    print("better model saved")
                print("current min_mae: {}({} sample)".format(1/self.MAX_MAEINV, num2ord(self.MAX_SAMPNUM)))
                print("current number of samples: {}".format(len(self.samples)))
                      
            except Exception as e:
                print(e)
                self.TASK_QUEUE.append(job)
                self.sample_nodes.append(sample_node)
                print("current queue length:", len(self.TASK_QUEUE))


    def search(self):

        while len(self.search_space) > 0:
            self.dump_all_states(len(self.samples))
            print("\niteration:", self.ITERATION)

            # evaluate jobs:
            print("\nevaluate jobs...")
            self.evaluate_jobs()
            print("\nfinished all jobs in task queue")

            # assemble the training data:
            print("\npopulate training data...")
            self.populate_training_data()
            print("finished")

            # training the tree
            print("\ntrain classifiers in nodes...")
            if torch.cuda.is_available():
                print("using cuda device")
            else:
                print("using cpu device")
            self.train_nodes()
            print("finished")
            self.print_tree()

            # clear the data in nodes
            print("\nclear training data...")
            self.reset_node_data()
            print("finished")

            print("\npopulate prediction data...")
            self.populate_prediction_data()
            print("finished")

            print("\npredict and partition nets in search space...")
            self.predict_nodes()
            self.check_leaf_bags()
            print("finished")
            self.print_tree()

            for i in range(0, 50):
                # select
                target_bin   = self.select()
                sampled_arch = target_bin.sample_arch()
                # NOTED: the sampled arch can be None
                if sampled_arch is not None:
                # TODO: back-propogate an architecture
                # push the arch into task queue
                    print("\nselected node" + str(target_bin.id-31) + " in leaf layer")
                    print("sampled arch:", sampled_arch)
                    if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                        self.TASK_QUEUE.append(sampled_arch)
                        self.search_space.remove(sampled_arch)
                        self.sample_nodes.append(target_bin.id-31)
                else:
                    # trail 1: pick a network from the best leaf
                    for n in self.nodes:
                        if n.is_leaf == True:
                            sampled_arch = n.sample_arch()
                            if sampled_arch is not None:
                                print("\nselected node" + str(n.id-31) + " in leaf layer")
                                print("sampled arch:", sampled_arch)
                                if json.dumps(sampled_arch) not in self.DISPATCHED_JOB:
                                    self.TASK_QUEUE.append(sampled_arch)
                                    self.search_space.remove(sampled_arch)
                                    self.sample_nodes.append(n.id-31)
                                    break
                            else:
                                continue

            self.ITERATION += 1


if __name__ == '__main__':
    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    with open('search_space', 'rb') as file:
        search_space = pickle.load(file)
    arch_code_len = len(search_space[0])
    print("\nthe length of architecture codes:", arch_code_len)
    print("total architectures:", len(search_space))

    if os.path.isfile('results.csv') == False:
        with open('results.csv', 'w+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow(['sample_id', 'arch_code', 'sample_node', 'val_loss', 'test_mae', 'test_corr',
                             'test_multi_acc', 'test_bi_acc', 'test_f1'])

    state_path = 'states'
    files = os.listdir(state_path)
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.nodes), "nodes")
        print("=====>loads:", len(agent.samples), "samples")
        print("=====>loads:", len(agent.DISPATCHED_JOB), "dispatched jobs")
        print("=====>loads:", len(agent.TASK_QUEUE), "task_queue jobs")
        agent.search()
    else:
        agent = MCTS(search_space, 6, arch_code_len)
        agent.search()
