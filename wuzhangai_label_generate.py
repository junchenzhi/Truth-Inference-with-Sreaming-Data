import numpy as np
import pickle
import pandas as pd
import argparse



parser = argparse.ArgumentParser(description="-----[Wuzhangai-truth-inference-generate-simulation-data]-----")
parser.add_argument("--num_classes", default=[3, 3], type=list, help="num_answers")
parser.add_argument("--new_stream_data", default=False, help="num_answers")
options = parser.parse_args()



class SyntheticData(object):
    def __init__(self, state=None):
        self.probas = False  # not yet
        self.Random_num = np.random.RandomState(None)
        if type(state) == str:
            with open(state, 'rb') as handle:
                aux = pickle.load(handle)  # read from file
                self.Random_num.set_state(aux)
        elif type(state) == tuple or type(state) == int:
            self.Random_num.set_state(state)
        self.init_state = self.Random_num.get_state()  # to replicate

    def set_probas(self, file_matrix):
        self.conf_matrix = np.asarray(file_matrix)
        self.probas = True



    def synthetic_annotate_data(self, truth, num_classes, deterministic=True, hard=True):

        if not self.probas:
            self.set_probas()

        synthetic_annotators_group = self.conf_matrix
        synthetic_annotators_group = np.asarray(synthetic_annotators_group)

        # workers_num = self.conf_matrix.shape[0]
        workers_list = ['worker_id_a', 'worker_id_b', 'worker_id_c']
        instances_list = []
        for i in range(20):
            sent = 'instance_id_' + str(i)
            instances_list.append(sent)

        synthetic_annotations = {}
        for i in range(truth.shape[0]):
            synthetic_annotations_instance = []
            for j in range(len(num_classes)):
                this_truth = truth[i,j]
                # synthetic_annotations_instance_task = []
                synthetic_annotations_instance_task = {}
                # for w in range(workers_num):
                for w in range(len(workers_list)):
                    if deterministic:
                        worker_transition = synthetic_annotators_group[w]
                        if hard:
                            sample_prob = worker_transition[j, this_truth, :]
                        yo = np.argmax(self.Random_num.multinomial(1, sample_prob))
                        # synthetic_annotations_instance_task.append(yo)
                        synthetic_annotations_instance_task[workers_list[w]] = yo
                synthetic_annotations_instance.append(synthetic_annotations_instance_task)
            # synthetic_annotations.append(synthetic_annotations_instance)
            # synthetic_annotations[str(i)] = synthetic_annotations_instance
            synthetic_annotations[instances_list[i]] = synthetic_annotations_instance

        return synthetic_annotations



    def synthetic_annotate_data_stream(self, truth, num_classes, deterministic=True, hard=True):

        if not self.probas:
            self.set_probas()

        synthetic_annotators_group = self.conf_matrix
        synthetic_annotators_group = np.asarray(synthetic_annotators_group)

        # workers_num = self.conf_matrix.shape[0]
        workers_list = ['worker_id_a', 'worker_id_b', 'worker_id_c', 'worker_id_d']
        instances_list = []
        instances_list.append('instance_id_' + str(2))
        instances_list.append('instance_id_' + str(11))
        for i in range(21, 41):
            sent = 'instance_id_' + str(i)
            instances_list.append(sent)

        synthetic_annotations = {}
        for i in range(truth.shape[0]):
            synthetic_annotations_instance = []
            for j in range(len(num_classes)):
                this_truth = truth[i,j]
                # synthetic_annotations_instance_task = []
                synthetic_annotations_instance_task = {}
                # for w in range(workers_num):
                for w in range(len(workers_list)):
                    if deterministic:
                        worker_transition = synthetic_annotators_group[w]
                        if hard:
                            sample_prob = worker_transition[j, this_truth, :]
                        yo = np.argmax(self.Random_num.multinomial(1, sample_prob))
                        # synthetic_annotations_instance_task.append(yo)
                        synthetic_annotations_instance_task[workers_list[w]] = yo
                synthetic_annotations_instance.append(synthetic_annotations_instance_task)
            # synthetic_annotations.append(synthetic_annotations_instance)
            # synthetic_annotations[str(i)] = synthetic_annotations_instance

            synthetic_annotations[instances_list[i]] = synthetic_annotations_instance

        return synthetic_annotations



def get_annotations(num_classes,train_file,stream=False):
    # Here we first obtain the ground truth
    truth = []

    df = pd.read_csv(train_file)
    for row in df.iterrows():
        truth_instance = []

        truth_instance.append(row[1]['Input.truth_1'])
        truth_instance.append(row[1]['Input.truth_2'])
        # truth_instance.append(row[1]['Input.truth_3'])

        truth_instance = np.array(truth_instance)
        truth.append(truth_instance)
    truth = np.array(truth)
    print('truth:', truth)

    # num_classes = [3, 3]
    B = np.asarray([  # confusion matrices of different workers
        [
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]],
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]],
        ],
        [
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]],
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]],
        ],
        [
            [[0.1, 0.8, 0.1],
            [0.3, 0.5, 0.2],
            [0.0, 0.9, 0.1]],  # biased for class 2
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]]
        ],
        [
            [[0.9, 0.0, 0.1],
             [0.3, 0.5, 0.2],
             [0.0, 0.9, 0.1]],
            [[0.9, 0.1, 0.0],
             [0.2, 0.6, 0.2],
             [0.1, 0.0, 0.9]]
        ]
    ])
    GenerateData = SyntheticData()
    GenerateData.set_probas(B)
    if stream==False:
        annotations = GenerateData.synthetic_annotate_data(truth, num_classes)
    else:
        annotations = GenerateData.synthetic_annotate_data_stream(truth, num_classes)
    return truth, annotations



if options.new_stream_data == False:
    truth, simulation_annotations = get_annotations(num_classes=options.num_classes,
                                                    train_file="data/wuzhangai_truth_init.csv",
                                                    stream=options.new_stream_data)
    np.save('./data/simulation_annotations_init.npy', simulation_annotations)
    np.save('./data/truth_init.npy', truth)
    print('annotations', simulation_annotations)
else:
    truth, simulation_annotations = get_annotations(num_classes=options.num_classes,
                                                    train_file="data/wuzhangai_truth_stream.csv",
                                                    stream=options.new_stream_data)
    np.save('./data/simulation_annotations_stream_1.npy', simulation_annotations)
    np.save('./data/truth_stream_1.npy', truth)
