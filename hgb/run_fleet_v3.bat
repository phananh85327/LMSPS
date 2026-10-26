@echo off
REM ==========================================================================
REM  Fleet v3: the rerun on the rebuilt AMAZON data (review parts 1-9).
REM  Same protocol as Fleet v2's final4 / final6 (run_fleet_v2.bat), except:
REM    - the rebuilt data/AMAZON (amazon_extract.py + dat_generator.py, part 1)
REM    - train_search.py's current defaults: alpha on the validation edges
REM      (part 2), fold-by-fold weight-step features (part 3), plain P-P / C-C
REM      relations (part 9)
REM    - --dropout 0.5 (part 3: without it the supernet overfits)
REM    - its own folder, logs\Fleet v3, so the Fleet v2 records stay intact
REM  A run whose JSON record already exists is skipped, so the script can be
REM  relaunched after an interruption (a run cut off mid-way has no JSON yet
REM  and starts again).
REM
REM  Usage:  run_fleet_v3.bat pilot | final4 | final6 | reselect | push | all
REM    pilot            optional, once before the fleets (~2 h): seed 1 twice with
REM                     --deterministic; compare_search_records.py must print
REM                     IDENTICAL (same seed -> same run on this machine)
REM    final4 / final6  30 searches each, seeds 1-30 (about an hour per run on
REM                     the old data; the rebuilt data is smaller)
REM    reselect         reselect_fleet.py per ns, delta 0.010 and 0.005 (the
REM                     sensitivity line); output also saved next to the runs.
REM                     Paste the printed amazon_lp_v3_f<ns>_a<i> entries (the
REM                     averaged one-SE sets, simplest first) into arch.py
REM    push             commit logs\Fleet v3 and push it to the current branch
REM                     (needs git on PATH and the branch tracking a remote)
REM    all              final4, push, final6, push, reselect, push
REM ==========================================================================
setlocal
cd /d "%~dp0"
REM --- conda: in a plain cmd on this machine `python` is the Windows-Store stub,
REM     so activate lmsps_env the same way run_fleet_v2.bat does. Another machine:
REM     set CONDA_ROOT to its conda folder before calling this script
if not defined CONDA_ROOT set CONDA_ROOT=E:\miniconda3
if /I not "%CONDA_DEFAULT_ENV%"=="lmsps_env" (
  call "%CONDA_ROOT%\Scripts\activate.bat"
  call conda activate lmsps_env
)
python -c "import sys; print('python:', sys.executable)"
set COMMON=--dataset AMAZON --task link_prediction --target-relation "U,P" --num-hops 5 --num_final 10 --split-seed 1 --tgt-ended-paths --half-feats --stage 150 --deterministic --eval-no-gumbel --eval-exhaustive-cap 1000
set FINAL=--dedupe-reversed --warmup-epochs 15 --dropout 0.5
set OUT=logs\Fleet v3
if not exist "%OUT%\final_ns4" mkdir "%OUT%\final_ns4"
if not exist "%OUT%\final_ns6" mkdir "%OUT%\final_ns6"
if not exist "%OUT%\pilot" mkdir "%OUT%\pilot"

if "%1"=="" goto usage
if /I "%1"=="pilot" goto pilot
if /I "%1"=="final4" goto final4
if /I "%1"=="final6" goto final6
if /I "%1"=="reselect" goto reselect
if /I "%1"=="push" goto push
if /I "%1"=="all" goto all
goto usage

:pilot
for %%p in (a b) do python train_search.py %COMMON% %FINAL% --ns 4 --search-json "%OUT%\pilot\pilot_%%p.json" --seeds 1 > "%OUT%\pilot\pilot_%%p.txt" 2>&1
python compare_search_records.py "%OUT%\pilot\pilot_a.json" "%OUT%\pilot\pilot_b.json"
findstr /C:"deterministic implementation" "%OUT%\pilot\pilot_*.txt" && echo   a non-deterministic op ran: see the lines above
goto end

:final4
for /L %%s in (1,1,30) do call :search 4 %%s
goto end

:final6
for /L %%s in (1,1,30) do call :search 6 %%s
goto end

REM :search <ns> <seed> -- one run, skipped if its JSON record exists
:search
set REC=%OUT%\final_ns%1\search_ns%1_seed%2
if exist "%REC%.json" (
  echo   skip ns %1 seed %2: record exists
  goto :eof
)
echo   ns %1 seed %2
python train_search.py %COMMON% %FINAL% --ns %1 --search-json "%REC%.json" --seeds %2 > "%REC%.txt" 2>&1
goto :eof

:reselect
REM the search_* glob keeps the reselect_d0xx.json summaries out of the input
for %%k in (4 6) do (
  python reselect_fleet.py "%OUT%\final_ns%%k\search_*.json" --delta 0.010 --prefix amazon_lp_v3_f%%k --out "%OUT%\final_ns%%k\reselect_d010.json" > "%OUT%\final_ns%%k\reselect_d010.txt" 2>&1
  python reselect_fleet.py "%OUT%\final_ns%%k\search_*.json" --delta 0.005 --prefix amazon_lp_v3_f%%k_d005 --out "%OUT%\final_ns%%k\reselect_d005.json" > "%OUT%\final_ns%%k\reselect_d005.txt" 2>&1
  echo ===== ns %%k, delta 0.010 =====
  type "%OUT%\final_ns%%k\reselect_d010.txt"
  echo.
)
echo (delta 0.005 sensitivity: reselect_d005.txt in the same folders)
goto end

:push
git add "%OUT%"
git commit -q -m "Fleet v3 logs (run_fleet_v3.bat)" || echo   nothing new to commit
git push || echo   push failed: check that git is on PATH and the branch tracks a remote
goto end

:all
call "%~f0" final4
call "%~f0" push
call "%~f0" final6
call "%~f0" push
call "%~f0" reselect
call "%~f0" push
goto end

:usage
echo usage: %~nx0 pilot ^| final4 ^| final6 ^| reselect ^| push ^| all
:end
endlocal
