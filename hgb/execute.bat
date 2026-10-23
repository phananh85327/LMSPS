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
if "%~1"=="" (
  call "%~dp0run_fleet_v2.bat" all
) else (
  call "%~dp0run_fleet_v2.bat" %*
)
