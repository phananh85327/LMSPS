@echo off
REM execute.bat -- superseded 2026-10-19. It used to hold its own copy of the
REM search command (last version = pilot B: --dedupe-reversed --warmup-epochs 15
REM --ns-search 8, seeds 21-30; those 10 runs are the ns-search-8 schedule
REM ablation, JSON records in logs\Fleet v2\ablation_ns8). To keep ONE definition
REM of the command, this file now just forwards to run_fleet_v2.bat, which also
REM activates lmsps_env. No argument = reselect (seconds, harmless); the 30-hour
REM fleets must be asked for explicitly.
REM
REM   execute.bat            -> run_fleet_v2.bat reselect
REM   execute.bat final4     -> run_fleet_v2.bat final4
REM   execute.bat final6     -> run_fleet_v2.bat final6
REM (Until 2026-10-26 no argument ran "all": 60 searches writing over the committed
REM Fleet v2 logs.) The rerun on the rebuilt data is run_fleet_v3.bat.
if "%~1"=="" (
  call "%~dp0run_fleet_v2.bat" reselect
) else (
  call "%~dp0run_fleet_v2.bat" %*
)
