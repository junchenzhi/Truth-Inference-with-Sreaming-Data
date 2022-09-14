import numpy as np
import torch.nn.functional as F
import torch
import os


class Inference_Wuzhangai(object):
    def __init__(self, params, annotations = None):
        self.params = params
        self.annotations = annotations


        self.num_classes = params['num_classes']
        self.mode = params['mode']


        self.solved_instances_annotations = {}
        self.solved_instances_id = []
        self.tobesolved_instances_annotations = {}
        self.tobesolved_instances_id = []
        self.old_workers = []

        self.init_infered_posterior = []
        self.all_instances_posterior = []
        self.all_workers_ability = None
        self.infered_posterior_old = None

        self.current_all_workers = set()
        self.worker_old_normalizer_all, self.worker_old_pi_all = None, None



    def merge_dict(self, x, y):
        for k, v in x.items():
            if k not in y.keys():
                y[k] += v
        return y

    def merge_dict_annotations(self, x, y):
        y_final = {}
        for k, v in y.items():
            # if this instance has 'history' information
            if k in x.keys():
                y_final[k] = []
                for j in range(len(v)):
                    y_final[k].append(self.merge_dict(x[k][j], y[k][j]))
            else:
                y_final[k] = v
        return y


    def unite_new_old_annotations(self, old_annotations, new_stream_annotations):
        self.annotations = self.merge_dict_annotations(old_annotations, new_stream_annotations)
        print('self.annotations', self.annotations)



    def to_onehot(self, lables):
        instance_num = len(lables[0])
        onehot_labels = []

        for i in range(len(self.num_classes)):
            onehot_labels_task = np.zeros((instance_num, self.num_classes[i]))
            for j in range(instance_num):
                onehot_labels_task[j, lables[i,j]] = 1
            onehot_labels.append(onehot_labels_task)
        return np.array(onehot_labels)


    def solve_kl(self, infered_posterior, truth):
        # truth = truth.reshape(truth.shape[1], truth.shape[0])
        # truth = to_onehot(truth, num_classes)

        # infered_posterior_agent = []
        # for j in range(len(infered_posterior)):
        #     infered_posterior_agent_task = []
        #     for key in infered_posterior[j]:
        #         infered_posterior_agent_task.append(infered_posterior[j][key])
        #
        #     infered_posterior_agent.append(np.array(infered_posterior_agent_task))
        # # infered_posterior_agent = np.array(infered_posterior_agent)
        #
        # truth_agent = []
        # for j in range(len(truth)):
        #     truth_task = []
        #     for key in truth[j]:
        #         truth_task.append(truth[j][key])
        #
        #     truth_agent.append(np.array(truth_task))
        # # truth_agent = np.array(truth_agent)
        #
        # infered_posterior_agent[infered_posterior_agent == 0] = 0.00001
        # truth_agent[truth_agent == 0] = 0.00001
        # kl_mean = F.kl_div(torch.tensor(np.log(infered_posterior_agent)), torch.tensor(truth_agent), reduction='mean')


        kl_mean = 0.0
        for j in range(len(infered_posterior)):
            infered_posterior_agent = []
            infered_posterior_agent_task = []
            for key in infered_posterior[j]:
                infered_posterior_agent_task.append(infered_posterior[j][key])

            infered_posterior_agent.append(np.array(infered_posterior_agent_task))
            infered_posterior_agent = np.array(infered_posterior_agent)

            truth_agent = []
            truth_task = []
            for key in truth[j]:
                truth_task.append(truth[j][key])

            truth_agent.append(np.array(truth_task))
            truth_agent = np.array(truth_agent)

            infered_posterior_agent[infered_posterior_agent == 0] = 0.00001
            truth_agent[truth_agent == 0] = 0.00001
            kl_mean += F.kl_div(torch.tensor(np.log(infered_posterior_agent)), torch.tensor(truth_agent),
                               reduction='mean')

        return kl_mean



    def test(self, infered_posterior, truth):
        acc_all_task = []

        infered_posterior_perpare = []
        for j in range(len(infered_posterior)):
            sent = []
            for _, value in infered_posterior[j].items():
                sent.append(value)
            infered_posterior_perpare.append(np.array(sent))
        infered_posterior_perpare = np.array(infered_posterior_perpare)

        for j in range(len(infered_posterior_perpare)):
            pred = np.argmax(infered_posterior_perpare[j], axis=1)
            acc = sum([1 if p == y else 0 for p, y in zip(pred, truth[:, j])]) / len(pred)
            acc_all_task.append(acc)

        return acc_all_task



    def inference_init(self, answers=None):
        if self.mode == 'train_stream':
            answers = self.annotations
        instance_num = len(answers)
        # Iterate tasks
        pred_all_task = []

        for j in range(len(self.num_classes)):
            pred = 1
            adjustment_factor = np.zeros((instance_num, self.num_classes[j]))
            i = 0
            for key, value in answers.items():
                for key_this in value[j].keys():
                    if value[j][key_this] != -1:
                        adjustment_factor[i, value[j][key_this]] += 1
                i += 1
            pred = adjustment_factor * pred
            pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
            pred_all_task.append(pred)

        for j in range(len(self.num_classes)):
            init_infered_posterior_task = {}
            i = 0
            for key, value in answers.items():
                init_infered_posterior_task[key] = pred_all_task[j][i]
                i += 1
            self.init_infered_posterior.append(init_infered_posterior_task)




    def e_step(self, answers, worker_ability):
        self.all_instances_posterior = []
        instance_num, worker_num = len(answers), len(self.current_all_workers)

        pred_all_task = []
        for j in range(len(self.num_classes)):
            pred = 1
            adjustment_factor = np.ones((instance_num, self.num_classes[j]))
            i = 0
            for _, value in answers.items():
                for w in self.current_all_workers:
                    if value[j][w] != -1:
                        adjustment_factor[i] *= worker_ability[j][w][:, value[j][w]]
                i += 1
            pred = adjustment_factor * pred
            pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
            pred_all_task.append(pred)

        for j in range(len(self.num_classes)):
            all_instances_posterior_task = {}
            i = 0
            for key, value in answers.items():
                all_instances_posterior_task[key] = pred_all_task[j][i]
                i += 1
            self.all_instances_posterior.append(all_instances_posterior_task)



    def m_step(self, truth, annotations):
        if self.mode == 'train_init':
            pi_all_task = []

            for j in range(len(self.num_classes)):
                pi = {}
                for w in self.current_all_workers:
                    pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                    normalizer = np.zeros(self.num_classes[j])
                    for id_key in truth[j].keys():
                        if annotations[id_key][j][w] != -1:
                            normalizer += truth[j][id_key]
                            pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
                    normalizer[normalizer == 0] = 0.00001
                    pi[w] = pi[w] / normalizer.reshape(self.num_classes[j], 1)
                pi_all_task.append(pi)


        else:
            pi_all_task = []
            for j in range(len(self.num_classes)):
                pi = {}
                for w in self.current_all_workers:
                    if w in self.worker_old_normalizer_all[j].keys():
                        normalizer = self.worker_old_normalizer_all[j][w]
                        pi[w] = self.worker_old_pi_all[j][w]
                    else:
                        normalizer = np.zeros(self.num_classes[j])
                        pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))

                    for id_key in truth[j]:
                        if annotations[id_key][j][w] != -1:
                            normalizer += truth[j][id_key]
                            pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]

                    normalizer[normalizer == 0] = 0.00001
                    pi[w] = pi[w] / normalizer.reshape(self.num_classes[j], 1)
                pi_all_task.append(pi)

        self.all_workers_ability = pi_all_task




    def m_step_save_information(self, truth, annotations):
        normalizer_all = []
        pi_all = []
        for j in range(len(self.num_classes)):
            pi, normalizer = {}, {}
            for w in self.current_all_workers:
                pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                normalizer[w] = np.zeros(self.num_classes[j])
                for id_key, id_value in annotations.items():
                    if id_value[j][w] != -1:
                        normalizer[w] += truth[j][id_key]
                        pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
            normalizer_all.append(normalizer)
            pi_all.append(pi)

        return normalizer_all, pi_all



    def compute_current_all_workers(self):
        for key, value in self.annotations.items():
            for i, item in enumerate(value):
                for sent in item.keys():
                    self.current_all_workers.add(sent)
        print('self.current_all_workers', self.current_all_workers)
        print('-------------------------------------------------------------------------------------------')





    def train(self, worker_old_normalizer_all=None, worker_old_pi_all=None, truth=None, flag = False):
        self.worker_old_normalizer_all = worker_old_normalizer_all
        self.worker_old_pi_all = worker_old_pi_all

        self.compute_current_all_workers()
        print('\nResults:')
        kl_divergence_divergence = [1.0] * self.params['patient']

        final_prediction_on_okinstances_probability, final_prediction_on_okinstances = {}, {}


        if flag != False:
            print('After initialization, the current ACC:', self.test(self.init_infered_posterior, truth))


        for epoch in range(self.params['epoch']):
            if epoch == 0:
                self.m_step(truth = self.init_infered_posterior, annotations = self.annotations)
            else:
                self.m_step(truth = self.all_instances_posterior, annotations = self.annotations)
            self.e_step(self.annotations, self.all_workers_ability)

            print('type(truth)', type(truth))



            if flag != False:
                print('Epoch time:', epoch + 1, '   ACC of all tasks:', self.test(self.all_instances_posterior, truth))
            else:
                # print('Epoch time:', epoch + 1, '   infered posterior:', self.all_instances_posterior)
                print('Epoch time:', epoch + 1)


            # Determine whether the calculation is converged, and determine which instances' inference results are credible
            if epoch != 0:
                kl_divergence_divergence_new_instance = self.solve_kl(self.all_instances_posterior, self.infered_posterior_old)
                kl_divergence_divergence.pop(0)
                kl_divergence_divergence.append(kl_divergence_divergence_new_instance)

                if min(kl_divergence_divergence) < self.params['kl_divergence_threshold']:

                    print('\nOK, stop inference!')
                    # iterate instance
                    for key in self.all_instances_posterior[0]:
                        over_flag = None
                        # Iterate instance's task
                        for j in range(len(self.num_classes)):
                            if (1 - max(self.all_instances_posterior[j][key])) < self.params['credible_instance_threshold']:
                                over_flag = True
                                break
                        if over_flag == True:
                            self.solved_instances_id.append(key)
                            self.solved_instances_annotations[key] = self.annotations[key]
                        else:
                            self.tobesolved_instances_id.append(key)
                            self.tobesolved_instances_annotations[key] = self.annotations[key]


                    okinstances_index_all_task = []
                    okinstances_probability_all_task = []
                    for j in range(len(self.num_classes)):
                        results_okinstances_probability = {item: self.all_instances_posterior[j][item] for item in self.solved_instances_id}
                        okinstances_index = {key: np.argmax(np.array(value)) for key, value in results_okinstances_probability.items()}
                        okinstances_index_all_task.append(okinstances_index)
                        okinstances_probability_all_task.append(results_okinstances_probability)

                    for i, item in enumerate(self.solved_instances_id):
                        final_prediction_on_okinstances[item] = []
                        for j in range(len(self.num_classes)):
                            final_prediction_on_okinstances[item].append(okinstances_index_all_task[j][item])
                            final_prediction_on_okinstances_probability[item] = okinstances_probability_all_task[j][item]


                    print('1) After analysis, the inference of these instances has been completed:\n', self.solved_instances_id)
                    print('\n')
                    print('Infered results (the following lists the results under different tasks):\n', final_prediction_on_okinstances)


                    print('\n')
                    print('2) These instances should require more review annotations in the future:', self.tobesolved_instances_id)
                    break

            self.infered_posterior_old = self.all_instances_posterior



        worker_normalizer_all, worker_pi_all = self.m_step_save_information(truth=self.all_instances_posterior, annotations=self.solved_instances_annotations)
        # 1 Save results
        np.save(os.path.join(self.params["result_path"], 'final_prediction_on_okinstances_probability.npy'), final_prediction_on_okinstances_probability)
        np.save(os.path.join(self.params["result_path"], 'final_prediction_on_okinstances.npy'), final_prediction_on_okinstances)

        # 2 Save annotations on the instances that still need to be annotated
        np.save(os.path.join(self.params["result_path"], 'tobesolved_instances_annotations.npy'), np.array(self.tobesolved_instances_annotations))

        # 3 Save worker capacity to await further stream calculations later
        np.save(os.path.join(self.params["result_path"], 'worker_normalizer_all.npy'), worker_normalizer_all)
        np.save(os.path.join(self.params["result_path"], 'worker_pi_all.npy'), worker_pi_all)

        print('\n')













