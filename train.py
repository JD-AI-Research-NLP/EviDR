import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--data_root_dir', type=str, default='numnet_plus_data')
    parser.add_argument('--data_dir', type=str, default='drop_dataset', help='')
    parser.add_argument('--code_dir', type=str, default='.')
    parser.add_argument('--max_epoch', type=str, default=10)
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--eval_batch_size', type=str, default=16)

    parser.add_argument('--SEED', type=int, default=345)
    parser.add_argument('--LR', type=float, default=5e-4)
    parser.add_argument('--BLR', type=float, default=1.5e-5)
    parser.add_argument('--WD', type=float, default=5e-5)
    parser.add_argument('--BWD', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")

    parser.add_argument('--add_relation_reasoning_module', default=False, action="store_true")
    parser.add_argument('--use_evidence_extractor', default=False, action="store_true")
    parser.add_argument('--use_gcn', default=False, action="store_true")
    parser.add_argument('--pretrain_model', type=str, default="roberta.base")
    parser.add_argument("--use_pretrained_model", default=False, action="store_true")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="squad_data/processed_data_electra.large/pretrained_model_squad/checkpoint_best.pt")
    parser.add_argument('--fp16', default=False, action='store_true')

    args, _ = parser.parse_known_args()

    DATA_ROOT_DIR = os.path.join('..', args.data_root_dir)
    DATA_DIR = os.path.join(DATA_ROOT_DIR, args.data_dir)
    CODE_DIR = args.code_dir

    MODEL_CONFIG = "--gcn_steps 3 "
    SAVE_DIR = "{}/numnet_plus_{}_LR_{}_BLR_{}_WD_{}_BWD_{}_BS_{}_gcn_{}_add_relation_module_{}".format(
        os.path.join(DATA_DIR, 'processed_data_' + args.pretrain_model), args.SEED, args.LR, args.BLR, args.WD,
        args.BWD, args.batch_size, args.use_gcn, args.add_relation_reasoning_module)
    DATA_CONFIG = "--data_dir {} --save_dir {}".format(os.path.join(DATA_DIR, 'processed_data_' + args.pretrain_model),
                                                       SAVE_DIR)
    TRAIN_CONFIG = "--batch_size {} --eval_batch_size {} --max_epoch {} --warmup 0.06 --optimizer adam \
                  --learning_rate {} --weight_decay {} --seed {} --gradient_accumulation_steps {} \
                  --bert_learning_rate {} --bert_weight_decay {} --log_per_updates 20 --eps 1e-6 --warmup_schedule {}".format(
        args.batch_size, args.eval_batch_size, args.max_epoch, args.LR, args.WD, args.SEED,
        args.gradient_accumulation_steps, args.BLR, args.BWD, args.warmup_schedule)

    BERT_CONFIG = "--pretrain_model {}".format(os.path.join(DATA_DIR, args.pretrain_model))

    print('start to train')
    train_cmd = ' '.join(
        ['python3', os.path.join(CODE_DIR, 'roberta_gcn_cli.py'), DATA_CONFIG, TRAIN_CONFIG, BERT_CONFIG,
         MODEL_CONFIG])
    if args.use_gcn:
        train_cmd = ' '.join([train_cmd, '--use_gcn'])

    if args.add_relation_reasoning_module:
        train_cmd = ' '.join([train_cmd, '--add_relation_reasoning_module'])
    if args.use_evidence_extractor:
        train_cmd = ' '.join([train_cmd, "--use_evidence_extractor"])
            
        
    if args.fp16:
        train_cmd = ' '.join([train_cmd, '--fp16'])

    if args.use_pretrained_model:
        pretrained_model_path = os.path.join(DATA_DIR, args.pretrained_model_path)
        train_cmd = ' '.join(
            [train_cmd, '--use_pretrained_model', '--pretrained_model_path {}'.format(pretrained_model_path)])

    print('execute cmd ', train_cmd)
    os.system(train_cmd)
