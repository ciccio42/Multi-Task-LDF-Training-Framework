# TO-DO

1) creare class RT1_video_cond
2) inserire RT1_video_cond nella pipeline di addestramento
3) verificare se multitask dataset va bene per l'addestramento (dimostratore panda + traiettorie pick-place)
4) addestra

## addestramenti
1° addestramento: \
  A) dim: panda sim  -- ur5 sim -> Tasso medio di successo (correlazione tra tasso medio e MSE) \
  B) test su istanze mai viste \
2° addestramento: \
  A) dim: panda sim -- ur5 reale FROM SCRATCH \
3° addestramento: \
  A) dim: panda sim -- ur5 reale FINETUNING dal checkpoint punto 1°