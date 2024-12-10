


from multi_task_il.datasets.command_encoder.command_encoder_dataset import TrajectoryCommandsDataset, CommandClassDefiner, ResultsDisplayer

import pickle

def main():
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    
    #TODO: refactory with hydra
    ROOT_PATH = '/user/frosa/multi_task_lfd/datasets/berkeley_autolab_ur5_converted/pick_place'
    OUTPUT_JSON_PATH = '.'
    PRINT_DATA_INFO = False

    # ce_dataset = TrajectoryCommandsDataset(ROOT_PATH)

    if PRINT_DATA_INFO:
        command_class_definer = CommandClassDefiner(ROOT_PATH)
        command_class_definer.produce_info(OUTPUT_JSON_PATH)


    dataset = TrajectoryCommandsDataset(root_path=ROOT_PATH,
                                        info_path='./dataset_info.json')
    
    for traj in dataset:
        print(traj)


    print('done')
    


if __name__ == '__main__':
    main()