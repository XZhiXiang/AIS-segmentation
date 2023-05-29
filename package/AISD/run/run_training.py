# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
import AISD
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from AISD.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from AISD.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import os


def main():
    parser = argparse.ArgumentParser()                                ##创建命令行选项、参数和子命令解析器
    parser.add_argument("-gpu", type=str, default='0')                ## 给parser实例增加属性   0是GPU的id   default 当参数未在命令行出现时使用的值

    parser.add_argument("-network", type=str, default='3d_fullres')
    parser.add_argument("-network_trainer", type=str, default='nnUNetTrainerV2_ResTrans')
    parser.add_argument("-task", type=str, default='1', help="can be task name or task id")    #可以是任务名或任务id
    parser.add_argument("-fold", type=str, default='all', help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-outpath", type=str, default='AISD', help='output path')       #输出路径
    parser.add_argument("-norm_cfg", type=str, default='IN', help='BN, IN or GN')              #归一化函数 默认为 IN ，可以是 BN，IN或GN
    parser.add_argument("-activation_cfg", type=str, default='LeakyReLU', help='LeakyReLU or ReLU')    #激活函数 LeakyReLU函数 或 ReLU函数

    parser.add_argument("-val", "--validation_only", default=False, help="use this if you want to only run the validation",
                        required=False, action="store_true")                         #仅运行validation时使用   action：当参数在命令行中出现时使用的动作基本类型
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)                             #设置使用压缩文件，使用解压文件需要更多的CPU和RAM
    parser.add_argument("--deterministic", default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")  #禁用混合精度训练并运行旧式fp32
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true')

    args = parser.parse_args()    # 属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu    # 设置当前使用的GPU设备

    norm_cfg = args.norm_cfg                      #归一化函数
    activation_cfg = args.activation_cfg          #激活函数
    outpath = args.outpath + '_' + norm_cfg + '_' + activation_cfg

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    if validation_only and (norm_cfg=='SyncBN'):
        norm_cfg='BN'

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage,\
    trainer_class = get_default_configuration(outpath, network, task, network_trainer, plans_identifier,
                                              search_in=(AISD.__path__[0], "training", "network_training"),
                                              base_module='AISD.training.network_training')

    trainer = trainer_class(plans_file, fold, norm_cfg, activation_cfg, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    if args.disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each 如果为false，则不会存储/覆盖最新的文件，但每个文件都是单独的
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest   是否将检查点保存为最新
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_latest_checkpoint(train=False)

        trainer.network.eval()          #测试/评估

        # predict validation            #预测
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)

        if network == '3d_lowres':
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
