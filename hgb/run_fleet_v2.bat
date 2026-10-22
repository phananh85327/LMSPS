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
REM  Usage:  run_fleet_v2.bat pilot | ns4 | ns6 | pilotB | final4 | final6 | reselect | analyze | all
REM    FINAL PROTOCOL (2026-10-18 evening):
REM    final4/final6  30 seeds each with --dedupe-reversed --warmup-epochs 15
REM            (NO --ns-search: width 8 made the evaluation tables less
REM            consistent across seeds, cross-seed gap 0.0085 -> 0.021). ~1 h/run.
REM            Output: logs\Fleet v2\final_ns4|final_ns6\*.txt + *.json
REM    reselect  1-SE re-selection over the saved evaluation tables
REM            (reselect_fleet.py, delta 0.010 = measured cross-seed noise; also
REM            prints delta 0.005 for the sensitivity line). Produces the per-run
REM            sets to report, the duplicate rate, the consensus and arch.py entries.
REM            Replayed on the first 10 old-schedule runs: argmin 10 distinct ->
REM            1-SE 3 distinct (x7 consensus).
REM    Earlier groups (kept as the schedule ablation; JSON records only, the txt
REM    logs were not kept): logs\Fleet v2\ablation_old55 = 55 candidates, no
REM    warm-up (10 runs + 1 same-seed duplicate); ablation_ns8 = dedupe + warm-up
REM    15 + ns-search 8 (10 runs). logs\Fleet v1 = the very first fleets
REM    (same seed in every run, 30 + 30 logs; GPU-nondeterminism study).
REM    pilot   two same-seed runs; PASS 2026-10-17 (identical).
REM    ns4/ns6 old-schedule targets, superseded by final4/final6.
REM    pilotB  the ns-search-8 pilot, superseded.
REM    analyze dedupe + path frequencies + consensus over raw argmin sets.
REM ==========================================================================
setlocal
cd /d "%~dp0"
REM --- conda: in a plain cmd on this machine `python` is the Windows-Store stub,
REM     so activate lmsps_env the same way execute.bat did (skip if already active)
if /I not "%CONDA_DEFAULT_ENV%"=="lmsps_env" (
  call E:\miniconda3\Scripts\activate.bat
  call conda activate lmsps_env
)
python -c "import sys; print('python:', sys.executable)"
set COMMON=--dataset AMAZON --task link_prediction --target-relation "U,P" --num-hops 5 --num_final 10 --split-seed 1 --tgt-ended-paths --half-feats --stage 150 --deterministic --eval-no-gumbel --eval-exhaustive-cap 1000
set FINAL=--dedupe-reversed --warmup-epochs 15
set OUT=logs\Fleet v2
if not exist "%OUT%\pilot" mkdir "%OUT%\pilot"
if not exist "%OUT%\ns4" mkdir "%OUT%\ns4"
if not exist "%OUT%\ns6" mkdir "%OUT%\ns6"
if not exist "%OUT%\pilotB" mkdir "%OUT%\pilotB"
if not exist "%OUT%\final_ns4" mkdir "%OUT%\final_ns4"
if not exist "%OUT%\final_ns6" mkdir "%OUT%\final_ns6"

if "%1"=="" goto usage
if /I "%1"=="pilot" goto pilot
if /I "%1"=="ns4" goto ns4
if /I "%1"=="ns6" goto ns6
if /I "%1"=="pilotB" goto pilotB
if /I "%1"=="final4" goto final4
if /I "%1"=="final6" goto final6
if /I "%1"=="reselect" goto reselect
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

:pilotB
for /L %%s in (21,1,25) do python train_search.py %COMMON% --ns 4 --dedupe-reversed --warmup-epochs 15 --ns-search 8 --search-json "%OUT%\pilotB\search_ns4_dd_w15_s8_seed%%s.json" --seeds %%s > "%OUT%\pilotB\search_ns4_dd_w15_s8_seed%%s.txt" 2>&1
echo.
echo Pilot B done. Compare with the same seeds under the old schedule:
echo   python analyze_search_fleet.py "%OUT%\pilotB\*.json" --prefix amazon_lp_v2b_s4
goto end

:analyze
python analyze_search_fleet.py "%OUT%\ns4\*.json" --prefix amazon_lp_v2_s4
echo.
python analyze_search_fleet.py "%OUT%\ns6\*.json" --prefix amazon_lp_v2_s6
goto end

:final4
for /L %%s in (23,1,23) do python train_search.py %COMMON% %FINAL% --ns 4 --search-json "%OUT%\final_ns4\search_ns4_seed%%s.json" --seeds %%s > "%OUT%\final_ns4\search_ns4_seed%%s.txt" 2>&1
goto end

:final6
for /L %%s in (13,1,13) do python train_search.py %COMMON% %FINAL% --ns 6 --search-json "%OUT%\final_ns6\search_ns6_seed%%s.json" --seeds %%s > "%OUT%\final_ns6\search_ns6_seed%%s.txt" 2>&1
goto end

:reselect
for %%k in (4 6) do (
  echo ===== ns %%k, delta 0.010 =====
  python reselect_fleet.py "%OUT%\final_ns%%k\*.json" --delta 0.010 --prefix amazon_lp_f%%k --out "%OUT%\final_ns%%k\reselect_d010.json"
  echo.
  echo ===== ns %%k, delta 0.005 ^(sensitivity^) =====
  python reselect_fleet.py "%OUT%\final_ns%%k\*.json" --delta 0.005 --prefix amazon_lp_f%%k_d005 --out "%OUT%\final_ns%%k\reselect_d005.json"
  echo.
)
goto end

:all
call "%~f0" final4
call "%~f0" final6
call "%~f0" reselect
goto end

:usage
echo usage: %~nx0 pilot ^| ns4 ^| ns6 ^| pilotB ^| final4 ^| final6 ^| reselect ^| analyze ^| all
:end
endlocal
