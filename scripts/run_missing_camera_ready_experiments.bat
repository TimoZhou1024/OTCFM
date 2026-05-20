@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Run the remaining camera-ready experiments called out by
REM reference/rebuttal_main_tex_gap_checklist.md.
REM
REM Usage:
REM   scripts\run_missing_camera_ready_experiments.bat
REM
REM Optional overrides, for example:
REM   set RUN_SENSITIVITY=0
REM   set DEVICE=cuda
REM   set SENS_DATASETS=scene15 bdgp cub nus_wide
REM   scripts\run_missing_camera_ready_experiments.bat

cd /d "%~dp0\.."

if not defined PY_CMD set "PY_CMD=uv run python"
if not defined DATA_ROOT set "DATA_ROOT=data"
if not defined DEVICE set "DEVICE="
if not defined STOP_ON_ERROR set "STOP_ON_ERROR=1"

if not defined RUN_ROBUSTNESS set "RUN_ROBUSTNESS=1"
if not defined RUN_ABLATION_IMVC set "RUN_ABLATION_IMVC=1"
if not defined RUN_ABLATION_MIXED set "RUN_ABLATION_MIXED=1"
if not defined RUN_SENSITIVITY set "RUN_SENSITIVITY=1"
if not defined RUN_RUNTIME set "RUN_RUNTIME=1"
if not defined RUN_CONVERGENCE set "RUN_CONVERGENCE=1"
if not defined RUN_TSNE set "RUN_TSNE=1"

if not defined BATCH_SIZE set "BATCH_SIZE=256"
if not defined ROBUST_EPOCHS set "ROBUST_EPOCHS=100"
if not defined ROBUST_RUNS set "ROBUST_RUNS=5"
if not defined ABLATION_EPOCHS set "ABLATION_EPOCHS=100"
if not defined ABLATION_RUNS set "ABLATION_RUNS=3"
if not defined SENS_EPOCHS set "SENS_EPOCHS=200"
if not defined SENS_RUNS set "SENS_RUNS=5"
if not defined RUNTIME_ITERATIONS set "RUNTIME_ITERATIONS=10"
if not defined RUNTIME_OPTION_EPOCHS set "RUNTIME_OPTION_EPOCHS=20"
if not defined RUNTIME_OPTION_PRETRAIN_EPOCHS set "RUNTIME_OPTION_PRETRAIN_EPOCHS=6"
if not defined CONVERGENCE_EPOCHS set "CONVERGENCE_EPOCHS=100"
if not defined TSNE_EPOCHS set "TSNE_EPOCHS=100"

if not defined ROBUSTNESS_DATASETS set "ROBUSTNESS_DATASETS=bdgp coil20 cub handwritten nus_wide scene15"
if not defined ABLATION_DATASETS set "ABLATION_DATASETS=scene15 bdgp coil20 cub handwritten nus_wide"
if not defined SENS_DATASETS set "SENS_DATASETS=scene15 bdgp coil20 cub handwritten nus_wide"
if not defined RUNTIME_DATASETS set "RUNTIME_DATASETS=nus-wide"
if not defined CONVERGENCE_DATASETS set "CONVERGENCE_DATASETS=bdgp nus_wide"
if not defined TSNE_DATASETS set "TSNE_DATASETS=coil20"

if not defined ROBUSTNESS_OUT set "ROBUSTNESS_OUT=results/robustness/camera_ready_full"
if not defined ABLATION_IMVC_OUT set "ABLATION_IMVC_OUT=results/ablation/camera_ready_imvc"
if not defined ABLATION_MIXED_OUT set "ABLATION_MIXED_OUT=results/ablation/camera_ready_mixed"
if not defined SENS_OUT_ROOT set "SENS_OUT_ROOT=sensitivity_results/camera_ready_full"
if not defined RUNTIME_OUT set "RUNTIME_OUT=benchmark_results/camera_ready_nus_wide"
if not defined CONVERGENCE_OUT_ROOT set "CONVERGENCE_OUT_ROOT=multi_seed_results/camera_ready"
if not defined TSNE_OUT set "TSNE_OUT=figures/camera_ready_tsne"

set "DEVICE_ARG="
if not "%DEVICE%"=="" set "DEVICE_ARG=--device %DEVICE%"

echo ================================================================================
echo Camera-ready missing experiment runner
echo ================================================================================
echo PY_CMD: %PY_CMD%
echo DATA_ROOT: %DATA_ROOT%
echo DEVICE: %DEVICE%
echo STOP_ON_ERROR: %STOP_ON_ERROR%
echo.
echo Phases:
echo   RUN_ROBUSTNESS=%RUN_ROBUSTNESS%
echo   RUN_ABLATION_IMVC=%RUN_ABLATION_IMVC%
echo   RUN_ABLATION_MIXED=%RUN_ABLATION_MIXED%
echo   RUN_SENSITIVITY=%RUN_SENSITIVITY%
echo   RUN_RUNTIME=%RUN_RUNTIME%
echo   RUN_CONVERGENCE=%RUN_CONVERGENCE%
echo   RUN_TSNE=%RUN_TSNE%
echo ================================================================================

if "%RUN_ROBUSTNESS%"=="1" (
    for %%D in (%ROBUSTNESS_DATASETS%) do (
        call :run_step "Full six-dataset robustness baselines: %%D" "%PY_CMD% scripts/run_robustness_test.py --test_type both --dataset %%D --data_root %DATA_ROOT% --epochs %ROBUST_EPOCHS% --batch_size %BATCH_SIZE% --num_runs %ROBUST_RUNS% --save_dir %ROBUSTNESS_OUT% --missing_rates 0.0 0.1 0.3 0.5 0.7 --unaligned_rates 0.0 0.2 0.4 0.6 %DEVICE_ARG%"
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%RUN_ABLATION_IMVC%"=="1" (
    for %%D in (%ABLATION_DATASETS%) do (
        call :run_step "Aligned incomplete component ablation: %%D" "%PY_CMD% scripts/run_ablation.py --dataset %%D --data_root %DATA_ROOT% --analysis component --modes full no_gw no_flow no_contrastive no_cluster no_recon --epochs %ABLATION_EPOCHS% --batch_size %BATCH_SIZE% --num_runs %ABLATION_RUNS% --missing_rate 0.7 --unaligned_rate 0.0 --save_dir %ABLATION_IMVC_OUT% %DEVICE_ARG%"
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%RUN_ABLATION_MIXED%"=="1" (
    for %%D in (%ABLATION_DATASETS%) do (
        call :run_step "Mixed missing+unaligned stress ablation: %%D" "%PY_CMD% scripts/run_ablation.py --dataset %%D --data_root %DATA_ROOT% --analysis component --modes full no_gw no_flow no_cluster no_recon --epochs %ABLATION_EPOCHS% --batch_size %BATCH_SIZE% --num_runs %ABLATION_RUNS% --missing_rate 0.7 --unaligned_rate 0.5 --save_dir %ABLATION_MIXED_OUT% %DEVICE_ARG%"
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%RUN_SENSITIVITY%"=="1" (
    for %%D in (%SENS_DATASETS%) do (
        call :run_step "Complete 9-parameter sensitivity: %%D" "%PY_CMD% scripts/run_sensitivity_analysis.py --dataset %%D --data_root %DATA_ROOT% --mode full --epochs %SENS_EPOCHS% --batch_size %BATCH_SIZE% --num_runs %SENS_RUNS% --output_dir %SENS_OUT_ROOT%/%%D %DEVICE_ARG%"
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%RUN_RUNTIME%"=="1" (
    call :run_step "NUS-WIDE non-diffusion runtime benchmark" "%PY_CMD% scripts/benchmark_comprehensive.py --datasets %RUNTIME_DATASETS% --benchmark_mode baselines --methods MRG-UMC CANDY SURE --data_dir %DATA_ROOT% --batch_size %BATCH_SIZE% --iterations %RUNTIME_ITERATIONS% --output_dir %RUNTIME_OUT% --option_epochs %RUNTIME_OPTION_EPOCHS% --option_pretrain_epochs %RUNTIME_OPTION_PRETRAIN_EPOCHS% --option_batch_size 128"
    if errorlevel 1 exit /b !errorlevel!
)

if "%RUN_CONVERGENCE%"=="1" (
    for %%D in (%CONVERGENCE_DATASETS%) do (
        call :run_step "Multi-seed convergence: %%D" "%PY_CMD% scripts/run_multi_seed_convergence.py --dataset %%D --epochs %CONVERGENCE_EPOCHS% --n_seeds 5 --start_seed 42 --output_dir %CONVERGENCE_OUT_ROOT%/%%D"
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%RUN_TSNE%"=="1" (
    for %%D in (%TSNE_DATASETS%) do (
        call :run_step "t-SNE visualization: %%D" "%PY_CMD% scripts/run_tsne_visualization.py --dataset %%D --epochs %TSNE_EPOCHS% --checkpoints 0,%TSNE_EPOCHS% --seed 42 --output_dir %TSNE_OUT% --batch_size %BATCH_SIZE%"
        if errorlevel 1 exit /b !errorlevel!
    )
)

echo.
echo ================================================================================
echo Requested experiment phases finished.
echo ================================================================================
exit /b 0

:run_step
set "STEP_NAME=%~1"
set "STEP_CMD=%~2"
echo.
echo ================================================================================
echo %STEP_NAME%
echo %STEP_CMD%
echo ================================================================================
call %STEP_CMD%
set "STEP_ERR=%ERRORLEVEL%"
if not "%STEP_ERR%"=="0" (
    echo ERROR: %STEP_NAME% failed with exit code %STEP_ERR%.
    if "%STOP_ON_ERROR%"=="1" exit /b %STEP_ERR%
)
exit /b 0
