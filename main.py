import numpy as np
import argparse
import os
from inference_class import Inference_Wuzhangai
import warnings







def main():
    parser = argparse.ArgumentParser(description="-----[Wuzhangai-truth-inference]-----")
    parser.add_argument("--mode", default="train_init", help="training mode (train_init/train_stream). If the algorithm is being run for the first time, the mode is 'train_init'; if new data is received and further inference is required, the mode is 'train_stream'")
    # parser.add_argument("--num_classes", default= [3,3], type=list, help="num_answers")
    parser.add_argument('--num_classes', nargs='+', help='<Required> Set flag', required=True)

    parser.add_argument('--result_path', default='./results/', type=str, help="result path")
    parser.add_argument("--epoch", default=20, type=int, help="number of epoch")
    parser.add_argument("--patient", default=5, type=int, help="patient number (epochs)")
    parser.add_argument("--kl_divergence_threshold", default=0.001, type=float, help="the KL divergence between the results of adjacent epochs that can be allowed")
    parser.add_argument("--credible_instance_threshold", default=0.05, type=float, help="the threshold for judging whether the inference result of the instance is credible")

    parser.add_argument('--init_data_path', default='./data/simulation_annotations_init.npy', type=str, help="result path")
    # parser.add_argument('--init_data_path', default='./data/initdata.npy', type=str,
    #                     help="result path")
    parser.add_argument('--new_data_path', default='./data/simulation_annotations_stream_1.npy', type=str, help="result path")
    parser.add_argument('--old_annotations_path', default='./results/tobesolved_instances_annotations.npy', type=str,
                        help="result path")
    parser.add_argument('--old_worker_ability_path_1', default='./results/worker_normalizer_all.npy', type=str,
                        help="result path")
    parser.add_argument('--old_worker_ability_path_2', default='./results/worker_pi_all.npy', type=str,
                        help="result path")
    parser.add_argument('--load_truth_information', default=False, help="load_truth_information")
    options = parser.parse_args()

    '''
    'num_classes' is the label categories of different questions.
    For example, if we have 2 questions for the crowdsourcing workers, 
    where the number of candidates for each question is 3 and 4, then 'num_classes' is '3,4'.
    '''

    num_classes = options.num_classes
    num_classes_list = []
    for i in range(len(num_classes)):
        if num_classes[i] != ',':
            num_classes_list.append(int(num_classes[i]))
    print('num_classes_list', num_classes_list)

    params = {
        'mode': options.mode,
        "num_classes": num_classes_list,
        "result_path": options.result_path,
        "epoch": options.epoch,
        "patient": options.patient,
        "kl_divergence_threshold": options.kl_divergence_threshold,
        "credible_instance_threshold": options.credible_instance_threshold,
    }
    if not os.path.exists(params["result_path"]):
        os.makedirs(params["result_path"])







    warnings.filterwarnings('ignore')
    truth = None
    flag = False
    if options.load_truth_information != False and options.mode == 'train_init':
        truth = np.load('./data/truth_init.npy')
        flag = True




    if options.load_truth_information != False and options.mode == 'train_stream':
        flag = True
        truth = np.load('./data/truth_stream_1.npy')




    if options.mode == "train_init":
        init_annotations = np.load(options.init_data_path, allow_pickle=True).item()
        model = Inference_Wuzhangai(params, init_annotations)
        print('\n')
        print('-------------------------------------------------------------------------------------------')
        print('init_annotations \n', init_annotations)
        print('-------------------------------------------------------------------------------------------')
        model.inference_init(init_annotations)
        model.train(truth=truth, flag=flag)

    else:
        new_stream_annotations = np.load(options.new_data_path, allow_pickle=True).item()
        old_annotations = np.load(options.old_annotations_path, allow_pickle=True).item()

        worker_old_normalizer_all = np.load(options.old_worker_ability_path_1, allow_pickle=True)
        worker_old_pi_all = np.load(options.old_worker_ability_path_2, allow_pickle=True)

        model = Inference_Wuzhangai(params)
        model.unite_new_old_annotations(old_annotations, new_stream_annotations)
        model.inference_init()
        model.train(worker_old_normalizer_all, worker_old_pi_all, truth=truth, flag=flag)




if __name__ == "__main__":
    main()