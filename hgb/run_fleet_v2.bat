@echo off
REM ==========================================================================
REM  Stress-test fleet v2: 30 searches at top-K 4 + 30 at top-K 6 (teacher's
REM  protocol, log #17) with the reproducibility fixes of 2026-10-16/17.
REM
REM  Fixed decisions (2026-10-17): --split-seed 1 (same split as the first
REM  fleet in logs\Top K 4|6 and the HAN-improve downstream runs; EVERY
REM  downstream consumer, incl. main_path.py, must use --split-seed 1 too),
REM  --stage 150 (same budget as the first fleet; the paper uses 200).
REM  --num_final 10 = paper's V = 2M for M = 4..6; C(10,4) = C(10,6) = 210
REM  subsets, all scored once (--eval-exhaustive-cap), noise-free weights
REM  (--eval-no-gumbel), deterministic kernels (--deterministic).
REM  Seeds 1..30 are RUN seeds: distinct seeds = the stability test;
REM  same-seed deterministic runs are copies and prove only reproducibility.
REM
REM  Usage:  run_fleet_v2.bat pilot | ns4 | ns6 | analyze | all
REM    pilot   two same-seed runs (~2 h). PASS = identical "Epoch N," lines and
REM            identical "paths" line in pilot_a.txt / pilot_b.txt, and no
REM            "does not have a deterministic implementation" warning.
REM            Run this BEFORE the 55-65 h fleet.
REM    ns4/ns6 30 runs each (~55-65 min per run).
REM    analyze dedupe + path frequencies + consensus + arch.py entries.
REM ==========================================================================
setlocal
cd /d "%~dp0"
set COMMON=--dataset AMAZON --task link_prediction --target-relation "U,P" --num-hops 5 --num_final 10 --split-seed 1 --tgt-ended-paths --half-feats --stage 150 --deterministic --eval-no-gumbel --eval-exhaustive-cap 1000
set OUT=logs\Fleet v2
if not exist "%OUT%\pilot" mkdir "%OUT%\pilot"
if not exist "%OUT%\ns4" mkdir "%OUT%\ns4"
if not exist "%OUT%\ns6" mkdir "%OUT%\ns6"

if "%1"=="" goto usage
if /I "%1"=="pilot" goto pilot
if /I "%1"=="ns4" goto ns4
if /I "%1"=="ns6" goto ns6
if /I "%1"=="analyze" goto analyze
if /I "%1"=="all" goto all
goto usage

:pilot
for %%p in (a b) do python train_search.py %COMMON% --ns 4 --search-json "%OUT%\pilot\pilot_%%p.json" --seeds 1 > "%OUT%\pilot\pilot_%%p.txt" 2>&1
echo.
echo Pilot done. Compare the two runs:
echo   fc "%OUT%\pilot\pilot_a.txt" "%OUT%\pilot\pilot_b.txt" ^| findstr /C:"Epoch " /C:"paths "
echo   findstr /C:"deterministic implementation" "%OUT%\pilot\pilot_*.txt"
echo (no fc differences in Epoch/paths lines and no findstr hit = PASS)
goto end

:ns4
for /L %%s in (1,1,30) do python train_search.py %COMMON% --ns 4 --search-json "%OUT%\ns4\search_ns4_seed%%s.json" --seeds %%s > "%OUT%\ns4\search_ns4_seed%%s.txt" 2>&1
goto end

:ns6
for /L %%s in (1,1,30) do python train_search.py %COMMON% --ns 6 --search-json "%OUT%\ns6\search_ns6_seed%%s.json" --seeds %%s > "%OUT%\ns6\search_ns6_seed%%s.txt" 2>&1
goto end

:analyze
python analyze_search_fleet.py "%OUT%\ns4\*.json" --prefix amazon_lp_v2_s4
echo.
python analyze_search_fleet.py "%OUT%\ns6\*.json" --prefix amazon_lp_v2_s6
goto end

:all
call "%~f0" ns4
call "%~f0" ns6
call "%~f0" analyze
goto end

:usage
echo usage: %~nx0 pilot ^| ns4 ^| ns6 ^| analyze ^| all
:end
endlocal
