import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args): 
    # 깊은 복사 시 복사된 객체는 내용적으로는 동일할 수 있지만 원본과는 다른 새로운 메모리 공간을 참조한다.
    seed_list = copy.deepcopy(args["seed"]) # 딕셔너리의 seed를 깊은 복사하여 seed_list에 저장
    device = copy.deepcopy(args["device"]) # 딕셔너리의 device를 깊은 복사하여 device에 저장

    for seed in seed_list: # 시드 리스트에 있는 시드를 반복
        args["seed"] = seed # 시드값 지정(현재 시드 리스트에는 1993만 있으므로 이 경우만 실행될 것임)
        args["device"] = device # 디바이스 지정(현재 디바이스 리스트에는 "0"만 있으므로 GPU 0만 실행될 것임)
        _train(args) # _train함수에 시드값과 디바이스를 지정한 딕셔너리를 다시 전달


def _train(args):
    # 초기 클래스 수(100)와 클래스 증가량(100)이 같다면 init_cls의 값을 0으로 하고 아니라면 100 
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    # logs_name = "logs/fetril/kohyoung/100/100"
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    # logs_name 경로가 존재하지 않으면
    if not os.path.exists(logs_name):
        os.makedirs(logs_name) # 생성

    # logfilename = "logs/fetril/kohyoung/100/100/train/1993/cosine_resnet18"
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"], # 모델명
        args["dataset"], # 데이터셋
        init_cls, # 초기 클래스 수
        args["increment"], # 학습 시 클래스 증가량
        args["prefix"], # 작업(train) 
        args["seed"], # 시드
        args["convnet_type"], # 합성곱 신경망 유형
    )

    # 로깅은 프로그램 실행 중 발생하는 이벤트 정보, 경고 또는 오류 메시지를 기록하고 추적
    logging.basicConfig(
        level=logging.INFO, # 로그 메시지의 중요도(DEBUG, INFO, WARNING, ERROR, CRITICAL 순으로 중요도가 있음)
        format="%(asctime)s [%(filename)s] => %(message)s", # 로그 메시지의 형식
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"), #로그 메시지를 파일에 저장
            logging.StreamHandler(sys.stdout), # 로그 메시지를 터미널 또는 콘솔에 출력
        ],
    )
    
    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager( # 데이터를 관리객체 생성
        args["dataset"], # kohyoung
        args["shuffle"], # false
        args["seed"], # 1993
        args["init_cls"], # 100
        args["increment"], # 100
    )
    # factory모듈은 객체 생성과 관련된 작업을 수행한다.
    model = factory.get_model(args["model_name"], args) # fetril모델 생성

    # cnn_curve와 nme_curve라는 딕셔너리를 생성 후 초기화
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        # 모델의 파라미터 수를 로그에 기록
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager) # 모델 증분학습
        cnn_accy, nme_accy = model.eval_task() # 모델 평가 결과를 각각 cnn_accy와 nme_accy에 저장
        model.after_task()

        if nme_accy is not None: # 모델 평가에서 nme_accy가 존재하면 cnn과 nme에 대한 로그를 처리하고
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else: # nme_accy가 더 이상 존재하지 않으면 cnn_accy에 대한 로그만 처리
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    
def _set_device(args): # 디바이스 지정하는 함수
    device_type = args["device"] # device_type에 device키의 value가 전달됨("0")
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else: # "0"으로 인해서
            device = torch.device("cuda:{}".format(device)) # device = "cuda:0"이 되고

        gpus.append(device) # gpus에 cuda:0 gpu가 추가됨

    args["device"] = gpus # cuda0이 device키의 value가 됨


def _set_random(): # 랜덤시드 지정하는 함수
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args): # 딕셔너리 출력
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
