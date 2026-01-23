@echo off
setlocal enabledelayedexpansion

:: --- CONFIGURATION ---
set CONFIG=configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml
set SPLIT_DIR=dataset/DPWH-sample-Dataset/train/kfold_splits
set IMG_ROOT=C:/Users/jazzb/rt-deterv2/RT-DETRv2/dataset/DPWH-sample-Dataset/train/

:: --- RUN THE FOLDS ---

for /l %%i in (1,1,5) do (
    echo ========================================
    echo STARTING TRAINING FOR FOLD %%i
    echo ========================================
    
    python tools/train.py -c %CONFIG% ^
      --output-dir output/fold_%%i ^
      -u train_dataloader.dataset.ann_file=%SPLIT_DIR%/fold_%%i_train.json ^
         train_dataloader.dataset.img_folder=%IMG_ROOT% ^
         val_dataloader.dataset.ann_file=%SPLIT_DIR%/fold_%%i_val.json ^
         val_dataloader.dataset.img_folder=%IMG_ROOT%
         
    echo FOLD %%i COMPLETED.
)

echo ========================================
echo ALL 5 FOLDS FINISHED!
echo ========================================
pause