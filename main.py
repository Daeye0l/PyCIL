import json
import argparse
from trainer import train # trainer파일에 있는 train함수 임포트

# python main.py --config=./exps_kohyoung/fetril.json
def main():
    # 커맨드라인으로 들어온 argument들을 파싱한 객체
    args = setup_parser().parse_args() 
    # --config로 전달된 ./exps_kohyoung/fetril.json을 로드하여 해당 파라미터들을 param에 저장
    param = load_json(args.config)
    args = vars(args)  # args에 {'config': './exps_kohyoung/fetril.json'}딕셔너리가 전달된 상태에서
    args.update(param)  # fetril.json에 있던 파라미터들도 딕셔너리에 추가됨.

    train(args) # 최종적으로 train함수에 딕셔너리 형태의 데이터가 전달됨.


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    # argparse는 커맨드라인으로 들어오는 argument를 파싱한다.
    # argparse객체를 생성하여 스크립트 설명을 써 놓음
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    # --config명령어의 default가 './exps/finetune.json' -> 우리는 --config = ./exp_kohyoung/fetril.json로 수행
    parser.add_argument('--config', type=str, default='./exps_kohyoung/fetril.json',
                        help='Json file of settings.')

    return parser # argparse객체 parser를 반환

# main이 다른 모듈에서 임포트되면서 실행되지 않게 함.
# 즉 main에서 파이썬 파일을 실행했을 때만 코드가 실행되도록 지정
if __name__ == '__main__':
    main()
