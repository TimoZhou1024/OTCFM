@echo off
REM Batch script for comprehensive sensitivity analysis across multiple datasets
REM Usage: run_all_sensitivity.bat

echo ================================================================================
echo OT-CFM Comprehensive Sensitivity Analysis
echo ================================================================================
echo.

REM Configuration
set EPOCHS=200
set N_RUNS=5
set DATASETS=Scene15 Handwritten NoisyMNIST

REM Note: Dataset names are case-insensitive and will be normalized

echo Starting sensitivity analysis with:
echo   Epochs: %EPOCHS%
echo   Runs per config: %N_RUNS%
echo   Datasets: %DATASETS%
echo.

REM Create output directory
if not exist sensitivity_results mkdir sensitivity_results

REM Single parameter analysis for lambda_gw
echo.
echo [1/3] Single Parameter Analysis: lambda_gw
echo ================================================================================
for %%D in (%DATASETS%) do (
    echo Running on %%D...
    uv run python scripts/run_sensitivity_analysis.py ^
        --dataset %%D ^
        --param lambda_gw ^
        --num_runs %N_RUNS% ^
        --epochs %EPOCHS% ^
        --output_dir sensitivity_results/%%D_lambda_gw
    
    if errorlevel 1 (
        echo ERROR: Failed on %%D
    ) else (
        echo ✓ Completed %%D
    )
    echo.
)

REM Grid search for lambda_gw vs lambda_cluster
echo.
echo [2/3] Grid Search: lambda_gw vs lambda_cluster
echo ================================================================================
for %%D in (%DATASETS%) do (
    echo Running on %%D...
    uv run python scripts/run_sensitivity_analysis.py ^
        --dataset %%D ^
        --param lambda_gw lambda_cluster ^
        --num_runs 3 ^
        --epochs %EPOCHS% ^
        --plot_3d ^
        --output_dir sensitivity_results/%%D_grid_gw_cluster
    
    if errorlevel 1 (
        echo ERROR: Failed on %%D
    ) else (
        echo ✓ Completed %%D
    )
    echo.
)

REM Full analysis (only on Scene15 due to computational cost)
echo.
echo [3/3] Full Analysis on Scene15
echo ================================================================================
echo Running comprehensive parameter sweep...
uv run python scripts/run_sensitivity_analysis.py ^
    --dataset Scene15 ^
    --mode full ^
    --num_runs 5 ^
    --epochs %EPOCHS% ^
    --output_dir sensitivity_results/Scene15_full_analysis

if errorlevel 1 (
    echo ERROR: Full analysis failed
) else (
    echo ✓ Completed full analysis
)

REM Summary
echo.
echo ================================================================================
echo Analysis Complete!
echo ================================================================================
echo Results saved to: sensitivity_results/
echo.
echo Generated outputs:
dir /B sensitivity_results
echo.
echo To view statistical reports:
echo   type sensitivity_results\*\statistical_report.txt
echo.
echo To view plots:
echo   start sensitivity_results\Scene15_lambda_gw\single_param_sweep.pdf
echo.
