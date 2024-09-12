#!/bin/bash

# source test_mosaic_cond_target_obj.sh nut_assembly

sbatch -w gnode02 test_mosaic_cond_target_obj.sh pick_place &
sbatch -w gnode02 test_mosaic_cond_target_obj.sh nut_assembly &
# sbatch -w gnode02 test_mosaic_cond_target_obj.sh button &
sbatch -w gnode02 test_mosaic_cond_target_obj.sh stack_block &
