#!/bin/bash

# source test_mosaic_cond_target_obj.sh nut_assembly

sbatch -w gnode02 test_mosaic_cond_target_obj.sh nut_assembly &
sbatch -w gnode04 test_mosaic_cond_target_obj.sh pick_place
